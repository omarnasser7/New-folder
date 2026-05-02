import streamlit as st
import base64
import json
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from PIL import Image
from openai import OpenAI

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Marketing Caption Generator",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'IBM Plex Sans Arabic', sans-serif;
}

/* RTL support */
.rtl { direction: rtl; text-align: right; }
.ltr { direction: ltr; text-align: left; }

/* Caption result boxes */
.caption-box {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #e0e0e0;
    font-size: 15px;
    line-height: 1.8;
    min-height: 120px;
    margin-top: 8px;
}
.caption-box-gen  { border-left: 4px solid #378ADD; }
.caption-box-str  { border-left: 4px solid #1D9E75; }

.badge-gen { background:#E6F1FB; color:#185FA5; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-str { background:#EAF3DE; color:#3B6D11; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* Prompt preview */
.prompt-preview {
    background: #f0f4f8;
    border-radius: 8px;
    padding: 1rem;
    font-size: 13px;
    font-family: monospace;
    white-space: pre-wrap;
    border: 1px solid #dde3ea;
    line-height: 1.7;
}

/* Metric card */
.metric-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}

/* Star rating display */
.stars { font-size: 20px; letter-spacing: 2px; }
.star-on  { color: #EF9F27; }
.star-off { color: #d1d5db; }
</style>
""", unsafe_allow_html=True)

# ─── Language / i18n ────────────────────────────────────────────────────────
T = {
    "en": {
        "app_title": "🛍️ AI Marketing Caption Generator",
        "app_sub": "Prompt Engineering Research — General vs. Structured Prompt Comparison",
        "sidebar_title": "⚙️ Settings",
        "lang_label": "Caption Language",
        "apikey_label": "OpenAI API Key",
        "apikey_help": "Your key stays in-session only — never stored.",
        "tab1": "🖼️ Generate",
        "tab2": "📊 Analysis",
        "tab3": "📦 Batch",
        "tab4": "📋 Export",
        "upload_header": "Product Image",
        "upload_prompt": "Upload a product image",
        "upload_types": "PNG, JPG, WEBP",
        "params_header": "Marketing Parameters",
        "product_label": "Product Name",
        "audience_label": "Target Audience",
        "tone_label": "Tone",
        "platform_label": "Platform",
        "usp_label": "Unique Selling Point (USP)",
        "prompts_header": "Prompt Preview",
        "tab_gen": "General Prompt (A)",
        "tab_str": "Structured Prompt (B)",
        "generate_btn": "▶ Generate Both Captions",
        "gen_caption": "Caption A — General Prompt",
        "str_caption": "Caption B — Structured Prompt",
        "rate_header": "Rate This Caption",
        "criteria": ["Persuasiveness", "Professionalism", "Audience Alignment", "Creativity"],
        "rating_btn": "Save Ratings",
        "analysis_header": "📊 Results Analysis",
        "no_ratings": "Generate captions and save ratings first.",
        "export_header": "📋 Export Results",
        "export_btn": "Download Results (JSON)",
        "export_csv": "Download Ratings (CSV)",
        "batch_header": "📦 Batch Experiment",
        "batch_add": "Add Product",
        "batch_run": "▶ Run Batch",
        "batch_empty": "Add at least one product to run a batch.",
        "err_nokey": "⚠️ Please enter your OpenAI API key in the sidebar.",
        "err_noimg": "⚠️ Please upload a product image.",
        "gen_prompt_text": "Describe the product in this image and write a compelling marketing caption for it.",
        "winner": "🏆 Better performing prompt",
        "avg_score": "Average Score",
        "copy_hint": "Copy the caption above manually with Ctrl+C / Cmd+C",
        "tones": ["persuasive", "professional", "casual", "luxury", "energetic"],
        "platforms": ["Instagram", "Facebook", "LinkedIn", "Twitter/X", "General"],
    },
    "ar": {
        "app_title": "🛍️ مولّد تعليقات تسويقية بالذكاء الاصطناعي",
        "app_sub": "بحث هندسة الموجّهات — مقارنة الموجّه العام بالموجّه المنظّم",
        "sidebar_title": "⚙️ الإعدادات",
        "lang_label": "لغة التعليق",
        "apikey_label": "مفتاح OpenAI API",
        "apikey_help": "يُستخدم المفتاح في الجلسة فقط — لا يُخزَّن.",
        "tab1": "🖼️ توليد",
        "tab2": "📊 التحليل",
        "tab3": "📦 دفعة",
        "tab4": "📋 تصدير",
        "upload_header": "صورة المنتج",
        "upload_prompt": "ارفع صورة المنتج",
        "upload_types": "PNG أو JPG أو WEBP",
        "params_header": "معاملات التسويق",
        "product_label": "اسم المنتج",
        "audience_label": "الجمهور المستهدف",
        "tone_label": "الأسلوب",
        "platform_label": "المنصة",
        "usp_label": "نقطة البيع الفريدة (USP)",
        "prompts_header": "معاينة الموجّه",
        "tab_gen": "الموجّه العام (A)",
        "tab_str": "الموجّه المنظّم (B)",
        "generate_btn": "▶ توليد كلا التعليقَين",
        "gen_caption": "التعليق A — الموجّه العام",
        "str_caption": "التعليق B — الموجّه المنظّم",
        "rate_header": "قيّم هذا التعليق",
        "criteria": ["الإقناع", "الاحترافية", "توافق الجمهور", "الإبداع"],
        "rating_btn": "حفظ التقييمات",
        "analysis_header": "📊 تحليل النتائج",
        "no_ratings": "قم بتوليد التعليقات وحفظ التقييمات أولاً.",
        "export_header": "📋 تصدير النتائج",
        "export_btn": "تنزيل النتائج (JSON)",
        "export_csv": "تنزيل التقييمات (CSV)",
        "batch_header": "📦 تجربة دفعية",
        "batch_add": "إضافة منتج",
        "batch_run": "▶ تشغيل الدفعة",
        "batch_empty": "أضف منتجاً واحداً على الأقل.",
        "err_nokey": "⚠️ يرجى إدخال مفتاح OpenAI API في الشريط الجانبي.",
        "err_noimg": "⚠️ يرجى رفع صورة المنتج.",
        "gen_prompt_text": "صف المنتج في هذه الصورة واكتب تعليقاً تسويقياً مقنعاً له باللغة العربية.",
        "winner": "🏆 الموجّه الأفضل أداءً",
        "avg_score": "متوسط الدرجات",
        "copy_hint": "انسخ التعليق أعلاه يدوياً باستخدام Ctrl+C",
        "tones": ["مقنع", "احترافي", "غير رسمي", "فاخر", "نشيط"],
        "platforms": ["Instagram", "Facebook", "LinkedIn", "Twitter/X", "عام"],
    }
}


def t(key):
    return T[st.session_state.lang][key]


# ─── Session state init ──────────────────────────────────────────────────────
for k, v in {
    "lang": "en",
    "caption_general": "",
    "caption_structured": "",
    "ratings": {},
    "all_ratings": [],
    "image_b64": None,
    "image_mime": "image/jpeg",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ─────────────────────────────────────────────────────────────────
def image_to_b64(img_bytes: bytes, mime: str) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def build_general_prompt(lang: str) -> str:
    return T[lang]["gen_prompt_text"]


def build_structured_prompt(params: dict, lang: str) -> str:
    p = params
    usp_line = (f"\nUSP: {p['usp']}" if p.get("usp") else "") if lang == "en" else \
               (f"\nنقطة البيع الفريدة: {p['usp']}" if p.get("usp") else "")
    if lang == "ar":
        return (
            f"أنت خبير تسويق رقمي محترف. بناءً على صورة المنتج المُقدَّمة، "
            f"اكتب تعليقاً تسويقياً مقنعاً باللغة العربية.\n\n"
            f"المنتج: {p['product_name']}\n"
            f"الجمهور المستهدف: {p['target_audience']}\n"
            f"الأسلوب: {p['tone']}\n"
            f"المنصة: {p['platform']}"
            f"{usp_line}\n\n"
            f"اكتب تعليقاً واحداً فعّالاً مع دعوة واضحة للتصرف (CTA) مناسب للمنصة المذكورة."
        )
    return (
        f"You are a professional digital marketing expert. "
        f"Based on the provided product image, write a persuasive marketing caption.\n\n"
        f"Product: {p['product_name']}\n"
        f"Target Audience: {p['target_audience']}\n"
        f"Tone: {p['tone']}\n"
        f"Platform: {p['platform']}"
        f"{usp_line}\n\n"
        f"Write a single effective caption with a clear CTA appropriate for the specified platform."
    )


def generate_caption(client: OpenAI, prompt: str, b64: str, mime: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.choices[0].message.content.strip()


def star_html(score: int) -> str:
    return "".join(
        f'<span class="star-on">★</span>' if i < score else f'<span class="star-off">★</span>'
        for i in range(5)
    )


def radar_chart(gen_scores: list, str_scores: list, criteria_labels: list):
    N = len(criteria_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    gen_vals = gen_scores + gen_scores[:1]
    str_vals = str_scores + str_scores[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, gen_vals, "o-", lw=2, color="#378ADD", label="General")
    ax.fill(angles_closed, gen_vals, alpha=0.15, color="#378ADD")
    ax.plot(angles_closed, str_vals, "o-", lw=2, color="#1D9E75", label="Structured")
    ax.fill(angles_closed, str_vals, alpha=0.15, color="#1D9E75")
    ax.set_xticks(angles)
    ax.set_xticklabels(criteria_labels, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
    ax.set_title("Criteria Comparison", size=11, pad=18, fontweight="medium")
    fig.tight_layout()
    return fig


def bar_chart(gen_scores: list, str_scores: list, criteria_labels: list):
    x = np.arange(len(criteria_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, gen_scores, w, label="General (A)", color="#378ADD", alpha=0.85)
    ax.bar(x + w / 2, str_scores, w, label="Structured (B)", color="#1D9E75", alpha=0.85)
    for xi, v in zip(x - w / 2, gen_scores):
        ax.text(xi, v + 0.05, str(v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, str_scores):
        ax.text(xi, v + 0.05, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_labels, fontsize=9)
    ax.set_ylim(0, 6)
    ax.set_ylabel("Score (1–5)")
    ax.axhline(3, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {t('sidebar_title')}")
    lang_choice = st.radio(t("lang_label"), ["English", "العربية"],
                           index=0 if st.session_state.lang == "en" else 1,
                           horizontal=True)
    st.session_state.lang = "en" if lang_choice == "English" else "ar"

    api_key = st.text_input(t("apikey_label"), type="password", help=t("apikey_help"))

    st.divider()
    st.caption("Model: GPT-4o Vision")
    st.caption("Evaluation: 5-point Likert Scale")
    st.caption("© Research Project — Prompt Engineering")


# ─── Header ──────────────────────────────────────────────────────────────────
rtl_cls = "rtl" if st.session_state.lang == "ar" else "ltr"
st.markdown(
    f'<div class="{rtl_cls}"><h2>{t("app_title")}</h2>'
    f'<p style="color:#666;font-size:14px;">{t("app_sub")}</p></div>',
    unsafe_allow_html=True,
)
st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([t("tab1"), t("tab2"), t("tab3"), t("tab4")])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    # ── Left: image + params ──────────────────────────────────
    with col_left:
        st.subheader(t("upload_header"))
        uploaded = st.file_uploader(
            t("upload_prompt"),
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            img_bytes = uploaded.read()
            mime_map = {"png": "image/png", "jpg": "image/jpeg",
                        "jpeg": "image/jpeg", "webp": "image/webp"}
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            st.session_state.image_mime = mime_map.get(ext, "image/jpeg")
            st.session_state.image_b64 = image_to_b64(img_bytes, st.session_state.image_mime)
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, use_container_width=True)

        st.divider()
        st.subheader(t("params_header"))
        product_name = st.text_input(
            t("product_label"),
            placeholder="e.g. Wireless Headphones" if st.session_state.lang == "en" else "مثال: سماعات لاسلكية"
        )
        target_audience = st.text_input(
            t("audience_label"),
            placeholder="e.g. Young professionals" if st.session_state.lang == "en" else "مثال: المهنيون الشباب"
        )
        col_t, col_p = st.columns(2)
        with col_t:
            tone = st.selectbox(t("tone_label"), t("tones"))
        with col_p:
            platform = st.selectbox(t("platform_label"), t("platforms"))
        usp = st.text_input(
            t("usp_label"),
            placeholder="e.g. 40-hour battery, noise cancellation" if st.session_state.lang == "en" else "مثال: بطارية 40 ساعة"
        )

    # ── Right: prompts + results ───────────────────────────────
    with col_right:
        params = {
            "product_name": product_name or ("the product" if st.session_state.lang == "en" else "المنتج"),
            "target_audience": target_audience or ("potential customers" if st.session_state.lang == "en" else "العملاء المحتملين"),
            "tone": tone,
            "platform": platform,
            "usp": usp,
        }
        gen_prompt = build_general_prompt(st.session_state.lang)
        str_prompt = build_structured_prompt(params, st.session_state.lang)

        st.subheader(t("prompts_header"))
        pt1, pt2 = st.tabs([t("tab_gen"), t("tab_str")])
        with pt1:
            st.markdown(f'<div class="prompt-preview">{gen_prompt}</div>', unsafe_allow_html=True)
        with pt2:
            st.markdown(f'<div class="prompt-preview">{str_prompt}</div>', unsafe_allow_html=True)

        st.divider()

        if st.button(t("generate_btn"), use_container_width=True, type="primary"):
            if not api_key:
                st.error(t("err_nokey"))
            elif not st.session_state.image_b64:
                st.error(t("err_noimg"))
            else:
                client = OpenAI(api_key=api_key)
                with st.spinner("Generating caption A..."):
                    try:
                        st.session_state.caption_general = generate_caption(
                            client, gen_prompt,
                            st.session_state.image_b64, st.session_state.image_mime
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                with st.spinner("Generating caption B..."):
                    try:
                        st.session_state.caption_structured = generate_caption(
                            client, str_prompt,
                            st.session_state.image_b64, st.session_state.image_mime
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")

        # ── Show captions ──
        if st.session_state.caption_general or st.session_state.caption_structured:
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<span class="badge-gen">A</span> <strong>{t("gen_caption")}</strong>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="caption-box caption-box-gen {rtl_cls}">'
                    f'{st.session_state.caption_general or "—"}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<span class="badge-str">B</span> <strong>{t("str_caption")}</strong>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="caption-box caption-box-str {rtl_cls}">'
                    f'{st.session_state.caption_structured or "—"}</div>',
                    unsafe_allow_html=True,
                )

            # ── Rating section ──
            st.divider()
            st.subheader(t("rate_header"))
            criteria = t("criteria")
            r_cols = st.columns(len(criteria))
            gen_scores, str_scores = [], []
            for i, (crit, col) in enumerate(zip(criteria, r_cols)):
                with col:
                    st.caption(crit)
                    g = st.slider(f"A – {crit}", 1, 5, 3, key=f"gen_{i}", label_visibility="collapsed")
                    s = st.slider(f"B – {crit}", 1, 5, 3, key=f"str_{i}", label_visibility="collapsed")
                    gen_scores.append(g)
                    str_scores.append(s)

            if st.button(t("rating_btn"), use_container_width=True):
                entry = {
                    "product": product_name,
                    "platform": platform,
                    "caption_general": st.session_state.caption_general,
                    "caption_structured": st.session_state.caption_structured,
                    "gen_scores": gen_scores,
                    "str_scores": str_scores,
                    "gen_avg": round(sum(gen_scores) / len(gen_scores), 2),
                    "str_avg": round(sum(str_scores) / len(str_scores), 2),
                }
                st.session_state.ratings = entry
                st.session_state.all_ratings.append(entry)
                st.success("✅ Ratings saved!")


# ════════════════════════════════════════════════════════════════
# TAB 2 — Analysis
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(t("analysis_header"))

    if not st.session_state.ratings:
        st.info(t("no_ratings"))
    else:
        r = st.session_state.ratings
        criteria = t("criteria")
        gen_scores = r["gen_scores"]
        str_scores = r["str_scores"]

        # ── Metric cards ──
        m1, m2, m3, m4 = st.columns(4)
        for col, label, gen, strd in zip(
            [m1, m2, m3, m4], criteria, gen_scores, str_scores
        ):
            with col:
                delta = strd - gen
                st.metric(label=label, value=f"{strd}/5", delta=f"{delta:+d} vs General")

        st.divider()

        # ── Charts ──
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_r = radar_chart(gen_scores, str_scores, criteria)
            st.pyplot(fig_r, use_container_width=True)
        with ch2:
            fig_b = bar_chart(gen_scores, str_scores, criteria)
            st.pyplot(fig_b, use_container_width=True)

        st.divider()

        # ── Summary table ──
        gen_avg = r["gen_avg"]
        str_avg = r["str_avg"]
        winner = "Structured (B)" if str_avg > gen_avg else ("General (A)" if gen_avg > str_avg else "Tie")
        diff = abs(str_avg - gen_avg)

        df_summary = pd.DataFrame({
            "Prompt": ["General (A)", "Structured (B)"],
            **{c: [gs, ss] for c, gs, ss in zip(criteria, gen_scores, str_scores)},
            t("avg_score"): [gen_avg, str_avg],
        })
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.success(f"{t('winner')}: **{winner}** (+{diff:.2f} points)")

        # ── All ratings aggregate ──
        if len(st.session_state.all_ratings) > 1:
            st.divider()
            st.subheader("All Sessions Aggregate")
            rows = []
            for entry in st.session_state.all_ratings:
                rows.append({
                    "Product": entry["product"],
                    "Platform": entry["platform"],
                    "Gen Avg": entry["gen_avg"],
                    "Str Avg": entry["str_avg"],
                    "Winner": "Structured" if entry["str_avg"] > entry["gen_avg"] else "General"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — Batch
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(t("batch_header"))
    st.caption("Define multiple products to generate and compare captions in bulk.")

    if "batch_products" not in st.session_state:
        st.session_state.batch_products = []

    with st.expander(f"➕ {t('batch_add')}", expanded=True):
        bc1, bc2 = st.columns(2)
        with bc1:
            b_name = st.text_input("Product Name", key="b_name")
            b_audience = st.text_input("Target Audience", key="b_audience")
            b_usp = st.text_input("USP", key="b_usp")
        with bc2:
            b_tone = st.selectbox("Tone", t("tones"), key="b_tone")
            b_platform = st.selectbox("Platform", t("platforms"), key="b_platform")
            b_url = st.text_input("Image URL (Unsplash/Pexels)", key="b_url",
                                  placeholder="https://images.unsplash.com/...")

        if st.button("Add to Batch"):
            if b_name and b_url:
                st.session_state.batch_products.append({
                    "product_name": b_name,
                    "target_audience": b_audience,
                    "tone": b_tone,
                    "platform": b_platform,
                    "usp": b_usp,
                    "image_url": b_url,
                })
                st.success(f"Added: {b_name}")
            else:
                st.warning("Product name and image URL are required.")

    if st.session_state.batch_products:
        st.write(f"**Queue:** {len(st.session_state.batch_products)} products")
        st.dataframe(
            pd.DataFrame(st.session_state.batch_products)[
                ["product_name", "platform", "tone", "usp"]
            ],
            use_container_width=True, hide_index=True,
        )

        if st.button(t("batch_run"), type="primary", use_container_width=True):
            if not api_key:
                st.error(t("err_nokey"))
            else:
                client = OpenAI(api_key=api_key)
                batch_results = []
                progress = st.progress(0)
                status = st.empty()
                for i, prod in enumerate(st.session_state.batch_products):
                    status.text(f"Processing {i+1}/{len(st.session_state.batch_products)}: {prod['product_name']}...")
                    try:
                        resp = requests.get(prod["image_url"], timeout=10)
                        mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
                        b64 = base64.b64encode(resp.content).decode("utf-8")
                        gp = build_general_prompt(st.session_state.lang)
                        sp = build_structured_prompt(prod, st.session_state.lang)
                        cap_g = generate_caption(client, gp, b64, mime)
                        cap_s = generate_caption(client, sp, b64, mime)
                        batch_results.append({
                            "Product": prod["product_name"],
                            "Platform": prod["platform"],
                            "Caption A (General)": cap_g,
                            "Caption B (Structured)": cap_s,
                        })
                    except Exception as e:
                        batch_results.append({
                            "Product": prod["product_name"],
                            "Platform": prod["platform"],
                            "Caption A (General)": f"ERROR: {e}",
                            "Caption B (Structured)": f"ERROR: {e}",
                        })
                    progress.progress((i + 1) / len(st.session_state.batch_products))
                status.text("✅ Batch complete!")
                st.session_state.batch_results = batch_results

        if "batch_results" in st.session_state and st.session_state.batch_results:
            df_batch = pd.DataFrame(st.session_state.batch_results)
            st.dataframe(df_batch, use_container_width=True, hide_index=True)
            csv = df_batch.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ Download Batch CSV", csv, "batch_captions.csv", "text/csv")
    else:
        st.info(t("batch_empty"))


# ════════════════════════════════════════════════════════════════
# TAB 4 — Export
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(t("export_header"))

    if not st.session_state.ratings:
        st.info(t("no_ratings"))
    else:
        r = st.session_state.ratings
        export_data = {
            "product": r.get("product", ""),
            "platform": r.get("platform", ""),
            "caption_general": r.get("caption_general", ""),
            "caption_structured": r.get("caption_structured", ""),
            "ratings": {
                "general": dict(zip(t("criteria"), r["gen_scores"])),
                "structured": dict(zip(t("criteria"), r["str_scores"])),
            },
            "averages": {
                "general": r["gen_avg"],
                "structured": r["str_avg"],
            },
        }
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.code(json_str, language="json")

        st.download_button(
            t("export_btn"),
            data=json_str,
            file_name="experiment_results.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.session_state.all_ratings:
            rows = []
            for entry in st.session_state.all_ratings:
                rows.append({
                    "Product": entry.get("product", ""),
                    "Platform": entry.get("platform", ""),
                    **{f"Gen_{c}": s for c, s in zip(t("criteria"), entry["gen_scores"])},
                    **{f"Str_{c}": s for c, s in zip(t("criteria"), entry["str_scores"])},
                    "Gen_Avg": entry["gen_avg"],
                    "Str_Avg": entry["str_avg"],
                })
            df_all = pd.DataFrame(rows)
            csv_all = df_all.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                t("export_csv"),
                data=csv_all,
                file_name="all_ratings.csv",
                mime="text/csv",
                use_container_width=True,
            )
