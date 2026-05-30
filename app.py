import os
import gc
import torch
import numpy as np
import pandas as pd
import requests
import base64
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

st.set_page_config(
    page_title="Image Captioning Refinement Fusion System",
    layout="wide",
    initial_sidebar_state="expanded"
)

JINA_KEY = os.environ.get("JINA_KEY", "")

JINA_URL     = "https://api.jina.ai/v1/rerank"
JINA_HEADERS = {
    "Authorization": f"Bearer {JINA_KEY}",
    "Content-Type":  "application/json"
}

if not JINA_KEY:
    st.error("JINA_KEY missing. Go to Space Settings → Secrets and add it.")
    st.stop()

@st.cache_resource
def load_local_models():
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
        BlipProcessor,
        BlipForImageTextRetrieval
    )
    gc.collect()

    florence_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    florence_model.eval()

    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    blip_itm_model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-large-coco",
        torch_dtype=torch.float32
    )
    blip_itm_model.eval()

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32
    )
    qwen_model.eval()

    return (
        florence_processor, florence_model,
        blip_processor, blip_itm_model,
        qwen_tokenizer, qwen_model
    )

def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def image_to_data_uri(image: Image.Image) -> str:
    raw = image_to_bytes(image)
    b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"

def generate_captions_florence(image: Image.Image, florence_proc, florence_mod) -> list:

    captions   = []
    image_size = (image.width, image.height)

    tasks = [
        ("<CAPTION>",               30,  {"num_beams": 1}),
        ("<CAPTION>",               35,  {"do_sample": True, "temperature": 1.0, "top_p": 0.92}),
        ("<DETAILED_CAPTION>",      80,  {"do_sample": True, "temperature": 0.7, "top_p": 0.90}),
        ("<DETAILED_CAPTION>",      90,  {"do_sample": True, "temperature": 1.1, "top_p": 0.95}),
        ("<MORE_DETAILED_CAPTION>", 120, {"do_sample": True, "temperature": 0.8, "top_p": 0.92}),
    ]

    for task_prompt, max_tokens, gen_params in tasks:
        try:
            inputs = florence_proc(
                text=task_prompt, images=image, return_tensors="pt"
            )
            with torch.no_grad():
                ids = florence_mod.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_tokens,
                    **gen_params
                )
            raw    = florence_proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = florence_proc.post_process_generation(
                raw, task=task_prompt, image_size=image_size
            )
            cap = parsed.get(task_prompt, "").strip().lower()
            captions.append(cap if cap else "a scene shown in the image")

        except Exception as e:
            st.warning(f"Florence {task_prompt} error: {str(e)[:80]}")
            captions.append("a scene shown in the image")

    seen, unique = set(), []
    for c in captions:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if len(unique) < 2:
        unique = captions

    while len(unique) < 5:
        unique.append(unique[0])

    return unique[:5]

def compute_itm_scores(image, captions, blip_proc, blip_itm) -> list:
    scores = []
    for cap in captions:
        try:
            inputs = blip_proc(
                images=image, text=cap,
                return_tensors="pt", padding=True
            )
            with torch.no_grad():
                out   = blip_itm(**inputs)
                score = torch.nn.functional.softmax(
                    out.itm_score, dim=1
                )[:, 1].item()
            scores.append(round(float(score), 4))
        except Exception as e:
            st.warning(f"ITM error: {str(e)[:60]}")
            scores.append(0.0)
    return scores

def compute_jina_scores(image: Image.Image, captions: list) -> list:
    img_data_uri = image_to_data_uri(image)
    scores       = []
    for cap in captions:
        try:
            payload = {
                "model":     "jina-reranker-m0",
                "query":     cap,
                "documents": [img_data_uri],
                "top_n":     1
            }
            response = requests.post(
                JINA_URL, headers=JINA_HEADERS,
                json=payload, timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if "results" in result and result["results"]:
                    score = result["results"][0].get("relevance_score", 0.0)
                    scores.append(round(float(score), 4))
                else:
                    scores.append(0.0)
            else:
                st.warning(f"Jina API error {response.status_code}: {response.text[:100]}")
                scores.append(0.0)
        except Exception as e:
            st.warning(f"Jina exception: {str(e)[:60]}")
            scores.append(0.0)
    return scores

def compute_cosine_scores(image, captions, blip_proc, blip_itm) -> list:
    try:
        img_inp = blip_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            vis      = blip_itm.vision_model(pixel_values=img_inp["pixel_values"])
            img_feat = blip_itm.vision_proj(vis.last_hidden_state[:, 0, :]).numpy()
            img_feat = normalize(img_feat, norm="l2")

        cap_inp = blip_proc(
            text=captions, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            txt      = blip_itm.text_encoder(
                input_ids=cap_inp["input_ids"],
                attention_mask=cap_inp["attention_mask"]
            )
            cap_feat = blip_itm.text_proj(txt.last_hidden_state[:, 0, :]).numpy()
            cap_feat = normalize(cap_feat, norm="l2")

        sims = cosine_similarity(img_feat, cap_feat)[0]
        return [round(float(s), 4) for s in sims]
    except Exception as e:
        st.warning(f"Cosine error: {str(e)[:60]}")
        return [0.0] * len(captions)

def majority_voting(captions, itm, jina, cosine) -> tuple:
    itm_r    = np.argsort(itm)[::-1]
    jina_r   = np.argsort(jina)[::-1]
    cosine_r = np.argsort(cosine)[::-1]

    votes = [
        int(itm_r[0]),    int(itm_r[1]),
        int(jina_r[0]),   int(jina_r[1]),
        int(cosine_r[0]), int(cosine_r[1])
    ]
    counts = Counter(votes)
    top2   = [idx for idx, _ in counts.most_common(2)]
    if len(top2) < 2:
        top2 = [int(itm_r[0]), int(jina_r[0])]

    return captions[top2[0]], captions[top2[1]], top2, dict(counts)

def fuse_captions(cap1: str, cap2: str, qwen_tok, qwen_mod) -> str:

    system_prompt = (
        "You write detailed image captions. "
        "You will receive two captions of the same image. "
        "Combine them into one complete, detailed caption. "
        "Include ALL visible details: "
        "clothing colors and style of each person, "
        "what each person looks like and what they are doing, "
        "objects and surroundings visible in the scene, "
        "and the background or setting. "
        "Write 3 to 4 complete sentences. "
        "Always finish the last sentence properly — never leave it incomplete. "
        "Use simple, clear, everyday words. "
        "Return ONLY the caption, nothing else."
    )

    user_prompt = (
        f"Caption A: {cap1}\n"
        f"Caption B: {cap2}\n\n"
        "Write a detailed caption covering all clothing, "
        "people, objects and background:"
    )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]

        text = qwen_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = qwen_tok([text], return_tensors="pt")

        with torch.no_grad():
            generated_ids = qwen_mod.generate(
                **model_inputs,
                max_new_tokens     = 220,
                do_sample          = False,
                repetition_penalty = 1.1,
                eos_token_id       = qwen_tok.eos_token_id,
                pad_token_id       = qwen_tok.eos_token_id
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        fused = qwen_tok.decode(output_ids, skip_special_tokens=True).strip()

        for prefix in ["Caption:", "Result:", "Answer:", "Fused caption:"]:
            if fused.lower().startswith(prefix.lower()):
                fused = fused[len(prefix):].strip()

        if fused and not fused.endswith((".", "!", "?")):
            last_stop = max(
                fused.rfind("."),
                fused.rfind("!"),
                fused.rfind("?")
            )
            if last_stop > len(fused) // 2:
                fused = fused[:last_stop + 1].strip()

        return fused if fused else cap1

    except Exception as e:
        st.warning(f"Qwen fusion error: {str(e)[:80]}")
        return cap1

def compute_caption_quality(image, final_caption, blip_proc, blip_itm) -> tuple:

    try:
        inputs = blip_proc(
            images=image, text=final_caption,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            out       = blip_itm(**inputs)
            itm_score = torch.nn.functional.softmax(
                out.itm_score, dim=1
            )[:, 1].item()
    except:
        itm_score = 0.0

    try:
        img_inp = blip_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            vis      = blip_itm.vision_model(pixel_values=img_inp["pixel_values"])
            img_feat = blip_itm.vision_proj(vis.last_hidden_state[:, 0, :]).numpy()
            img_feat = normalize(img_feat, norm="l2")

        cap_inp = blip_proc(
            text=[final_caption], return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            txt      = blip_itm.text_encoder(
                input_ids=cap_inp["input_ids"],
                attention_mask=cap_inp["attention_mask"]
            )
            cap_feat = blip_itm.text_proj(txt.last_hidden_state[:, 0, :]).numpy()
            cap_feat = normalize(cap_feat, norm="l2")

        cosine_score = float(cosine_similarity(img_feat, cap_feat)[0][0])
    except:
        cosine_score = 0.0

    avg_score = round((itm_score + cosine_score) / 2, 4)
    return avg_score, round(itm_score, 4), round(cosine_score, 4)

def render_gauge(score, itm, cosine, placeholder):

    if score >= 0.75:
        label, bar_color = "Good",     "#16a34a"
    elif score >= 0.50:
        label, bar_color = "Moderate", "#d97706"
    elif score >= 0.25:
        label, bar_color = "Low",      "#ca8a04"
    else:
        label, bar_color = "Poor",     "#dc2626"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        number = {
            "font":   {"size": 36, "color": bar_color, "family": "Arial Black"},
            "suffix": ""
        },
        gauge = {
            "axis": {
                "range":     [0, 1],
                "tickwidth": 2,
                "tickcolor": "#111827",
                "tickfont":  {"size": 11, "color": "#374151"}
            },
            "bar": {
                "color":     "#111827",
                "thickness": 0.06
            },
            "bgcolor":      "white",
            "borderwidth":  0,
            "steps": [
                {"range": [0.00, 0.25], "color": "#ef4444"},
                {"range": [0.25, 0.50], "color": "#f59e0b"},
                {"range": [0.50, 0.75], "color": "#84cc16"},
                {"range": [0.75, 1.00], "color": "#22c55e"},
            ],
            "threshold": {
                "line":      {"color": "#111827", "width": 5},
                "thickness": 0.85,
                "value":     score
            }
        },
        title = {
            "text": f"Caption Quality Score<br><b style='color:{bar_color};font-size:15px'>{label}</b>",
            "font": {"size": 13, "color": "#374151"}
        }
    ))

    fig.update_layout(
        height        = 240,
        margin        = dict(l=15, r=15, t=55, b=5),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = {"color": "#374151", "family": "Arial"}
    )

    with placeholder:
        st.markdown("<br>", unsafe_allow_html=True)
        g_col, s_col = st.columns([3, 2])

        with g_col:
            st.plotly_chart(fig, use_container_width=True)

        with s_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("**Score Breakdown**")
            st.markdown(f"Image-Text Match: **{itm}**")
            st.markdown(f"Embedding Similarity: **{cosine}**")
            st.markdown(f"Overall Score: **{score} / 1.00**")
            st.markdown(
                f"<span style='background:{bar_color};color:white;"
                f"padding:4px 12px;border-radius:12px;"
                f"font-weight:700;font-size:13px;'>{label}</span>",
                unsafe_allow_html=True
            )

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Image Captioning Refinement Fusion")
    st.markdown("---")
    st.markdown("### Pipeline Steps")
    st.markdown("""
**1. Florence-2-Large** (Local)
Generate 5 captions

**2. BLIP ITM** (Local)
Image-text matching

**3. Jina Reranker M0** (API)
Semantic reranking

**4. Cosine Similarity** (Local)
Embedding similarity

**5. Majority Voting**
Best 2 captions selected

**6. Qwen2.5-1.5B** (Local)
Caption fusion
    """)
    st.markdown("---")
    st.markdown("**Local:** Florence-2, BLIP ITM, Qwen2.5")
    st.markdown("**API:** Jina")
    st.markdown("---")
    st.markdown("### Caption Quality Metrics")
    st.markdown("""
**BLIP ITM** (Image-Text Match)
Measures how well the caption
matches the image content.

**Cosine Similarity**
Measures embedding distance
between image and caption.
    """)

# ── MAIN UI ────────────────────────────────────────────────────────
st.title("Image Captioning Refinement Fusion System")
st.markdown("Upload an image to generate a refined, grounded caption.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Select an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")

    col_img, col_run = st.columns([1, 1])

    with col_img:
        st.image(input_image, caption="Uploaded Image", use_container_width=True)
        gauge_placeholder = st.empty()

    with col_run:
        if st.button("Generate Caption", type="primary", use_container_width=True):

            st.info(" First time? Models are downloading and loading — this takes 3 to 4 minutes only once. After that, every run will be much faster!")
            st.warning(" Please do not refresh or close the page while models are loading.")

            with st.spinner("⏳ Loading Florence-2, BLIP and Qwen models... Please wait patiently!"):
                (
                    florence_proc, florence_mod,
                    blip_proc, blip_itm,
                    qwen_tok, qwen_mod
                ) = load_local_models()

            st.success(" All models loaded successfully! Starting the pipeline now...")

            progress = st.progress(0)
            status   = st.empty()

            status.info("Step 1/6: Generating captions with Florence-2-Large...")
            st.info(" Florence-2 is reading your image carefully — generating 5 diverse captions. This takes about 20–30 seconds, please hold on!")
            captions = generate_captions_florence(
                input_image, florence_proc, florence_mod
            )
            progress.progress(16)

            with st.expander("5 Generated Captions", expanded=True):
                for i, cap in enumerate(captions):
                    st.write(f"**{i+1}.** {cap}")

            status.info("Step 2/6: Computing BLIP ITM scores...")
            itm_scores = compute_itm_scores(
                input_image, captions, blip_proc, blip_itm
            )
            progress.progress(32)

            status.info("Step 3/6: Computing Jina Reranker scores...")
            jina_scores = compute_jina_scores(input_image, captions)
            progress.progress(50)

            status.info("Step 4/6: Computing Cosine Similarity scores...")
            cosine_scores = compute_cosine_scores(
                input_image, captions, blip_proc, blip_itm
            )
            progress.progress(66)

            scores_df = pd.DataFrame({
                "Caption": [f"Cap {i+1}: {c[:50]}" for i, c in enumerate(captions)],
                "ITM":     itm_scores,
                "Jina":    jina_scores,
                "Cosine":  cosine_scores
            })
            with st.expander("All Scores", expanded=False):
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

            status.info("Step 5/6: Running majority voting...")
            best_1, best_2, _, _ = majority_voting(
                captions, itm_scores, jina_scores, cosine_scores
            )
            progress.progress(83)

            st.markdown("### Majority Voted Captions")
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"1. {best_1}")
            with c2:
                st.info(f"2. {best_2}")

            status.info("Step 6/6: Fusing captions with Qwen2.5-1.5B...")
            st.success(" Best captions selected! Now crafting your final detailed caption — please hold on for about 30–60 seconds, it is definitely on the way!")
            final = fuse_captions(best_1, best_2, qwen_tok, qwen_mod)
            progress.progress(100)
            status.success("Pipeline complete!")

            st.markdown("---")
            st.markdown("### Final Fused Caption")
            st.markdown(
                f"<div style='"
                f"background:linear-gradient(135deg,#667eea,#764ba2);"
                f"padding:24px;border-radius:12px;color:white;"
                f"font-size:18px;font-weight:500;text-align:center;"
                f"line-height:1.6;'>{final}</div>",
                unsafe_allow_html=True
            )

            avg_score, itm_q, cosine_q = compute_caption_quality(
                input_image, final, blip_proc, blip_itm
            )

            st.session_state.avg_score = avg_score
            st.session_state.itm_q     = itm_q
            st.session_state.cosine_q  = cosine_q

            render_gauge(avg_score, itm_q, cosine_q, gauge_placeholder)