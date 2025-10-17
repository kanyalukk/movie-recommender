import pickle
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Recommender (Collaborative Filtering)")
st.caption(
    "อัปโหลดไฟล์ pickle ที่มี (user_similarity_df, user_movie_ratings) แล้วเลือก User ID เพื่อรับคำแนะนำภาพยนตร์ — ใช้ฟังก์ชัน get_movie_recommendations ที่คุณมีอยู่แล้ว"
)

# === Helpers ===
@st.cache_data(show_spinner=False)
def _load_pickle_from_path(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_pickle_from_bytes(b: bytes):
    # ไม่ cache เพื่อหลีกเลี่ยงการถือ reference กับ UploadedFile
    return pickle.loads(b)


def _infer_user_ids(user_movie_ratings):
    """พยายามเดาว่ามี user IDs อะไรบ้างจากโครงสร้างข้อมูลทั่วไป"""
    ids = []
    try:
        if isinstance(user_movie_ratings, pd.DataFrame):
            if "user_id" in user_movie_ratings.columns:
                ids = sorted(pd.unique(user_movie_ratings["user_id"]).tolist())
            else:
                # สมมติว่า index คือ user id
                ids = sorted(pd.unique(user_movie_ratings.index).tolist())
        elif isinstance(user_movie_ratings, dict):
            ids = sorted(list(user_movie_ratings.keys()))
    except Exception:
        pass
    return ids


# === Import your recommender function ===
try:
    from myfunction_66130701925 import get_movie_recommendations
    fn_loaded_ok = True
except Exception as e:
    fn_loaded_ok = False
    st.error(
        "ไม่สามารถ import ฟังก์ชัน get_movie_recommendations จากโมดูล `myfunction_66130701925` ได้\n"
        "➡️ วางไฟล์ `myfunction_66130701925.py` ไว้ในโฟลเดอร์เดียวกับแอปนี้แล้วลอง rerun"
    )
    with st.expander("รายละเอียดข้อผิดพลาด"):
        st.exception(e)


# === Sidebar: Data Source ===
st.sidebar.header("📦 Data Source (.pkl)")
source = st.sidebar.radio(
    "เลือกวิธีโหลดข้อมูล:",
    ("Upload .pkl", "ใช้ไฟล์โลคอลชื่อ recommendation_data.pkl"),
    index=0,
)

session_has_data = False

if source == "Upload .pkl":
    uploaded = st.sidebar.file_uploader("อัปโหลดไฟล์ .pkl", type=["pkl", "pickle"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            data_bytes = uploaded.read()
            user_similarity_df, user_movie_ratings = _load_pickle_from_bytes(data_bytes)
            st.session_state["data"] = (user_similarity_df, user_movie_ratings)
            session_has_data = True
            st.sidebar.success("โหลดข้อมูลจากไฟล์อัปโหลดแล้ว ✔")
        except Exception as e:
            st.sidebar.error("โหลดไฟล์ไม่สำเร็จ — ตรวจสอบว่าไฟล์ประกอบด้วย tuple: (user_similarity_df, user_movie_ratings)")
            with st.sidebar.expander("รายละเอียดข้อผิดพลาด"):
                st.sidebar.exception(e)
else:
    # local path
    try:
        user_similarity_df, user_movie_ratings = _load_pickle_from_path("recommendation_data.pkl")
        st.session_state["data"] = (user_similarity_df, user_movie_ratings)
        session_has_data = True
        st.sidebar.success("โหลดข้อมูลจากไฟล์โลคอลแล้ว ✔")
    except Exception as e:
        st.sidebar.warning("ไม่พบหรือโหลดไฟล์โลคอลไม่ได้ — ลองใช้วิธี Upload .pkl แทน")
        with st.sidebar.expander("รายละเอียดข้อผิดพลาด"):
            st.sidebar.exception(e)


if session_has_data:
    user_similarity_df, user_movie_ratings = st.session_state["data"]

    with st.expander("🔍 ดูตัวอย่างข้อมูล (Debug)", expanded=False):
        st.write("**ชนิดของ** user_similarity_df:", type(user_similarity_df))
        try:
            st.dataframe(getattr(user_similarity_df, "head", lambda: user_similarity_df)(), use_container_width=True)
        except Exception:
            st.write("ไม่สามารถแสดงตัวอย่าง user_similarity_df ได้")

        st.write("**ชนิดของ** user_movie_ratings:", type(user_movie_ratings))
        try:
            if isinstance(user_movie_ratings, pd.DataFrame):
                st.dataframe(user_movie_ratings.head(), use_container_width=True)
            else:
                st.code(str(user_movie_ratings)[:1500])
        except Exception:
            st.write("ไม่สามารถแสดงตัวอย่าง user_movie_ratings ได้")

    # --- Controls ---
    ids = _infer_user_ids(user_movie_ratings)
    col1, col2 = st.columns([2, 1])

    if ids:
        user_id = col1.selectbox("เลือก User ID", ids, index=0)
    else:
        user_id = col1.number_input("ใส่ User ID", value=1, step=1)

    k = int(col2.slider("จำนวนคำแนะนำ (Top-K)", min_value=5, max_value=50, value=10, step=1))

    run = st.button("✨ สร้างคำแนะนำ (Recommend)", type="primary", disabled=not fn_loaded_ok)

    if run:
        try:
            recs = get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, k)
            if not isinstance(recs, (list, tuple)):
                recs = list(recs) if recs is not None else []

            df_out = pd.DataFrame({
                "rank": list(range(1, len(recs) + 1)),
                "movie_title": recs,
            })

            st.success(f"Top {len(recs)} movie recommendations for User {user_id}")
            st.dataframe(df_out, use_container_width=True)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ ดาวน์โหลดผลลัพธ์เป็น CSV",
                data=csv_bytes,
                file_name=f"recommendations_user_{user_id}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error("คำนวณคำแนะนำไม่สำเร็จ — ตรวจสอบรูปแบบข้อมูลหรือฟังก์ชัน get_movie_recommendations อีกครั้ง")
            st.exception(e)
else:
    st.info("โปรดอัปโหลดไฟล์ .pkl หรือใช้ไฟล์โลคอลเพื่อเริ่มต้นใช้งานแอป")


# === Sidebar: About / Tips ===
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Tips")
st.sidebar.markdown(
    """
- โครงสร้างไฟล์ pickle ควรเป็น **tuple**: `(user_similarity_df, user_movie_ratings)`
- ฟังก์ชัน `get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, k)` จะถูกเรียกโดยตรงจากโมดูลของคุณ
- ถ้าแอปหา User IDs ไม่เจอ คุณสามารถพิมพ์ค่า `User ID` เองได้
- กดปุ่ม **Download CSV** เพื่อบันทึกผลลัพธ์
    """
)
