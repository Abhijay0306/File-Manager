import os
import dropbox
import streamlit as st
from google.cloud import vision
from docx import Document
import fitz  # PyMuPDF
from io import BytesIO
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import tempfile
import cv2
import imageio
import mimetypes


# --- CONFIGURATION ---
DROPBOX_FOLDER_PATH = '/Testing'  # Set to your folder, use "" for root
GOOGLE_CREDENTIALS_PATH = "client_secret_193101800486-1o170ie76co1t6pf6ojnb21u22kbfoak.apps.googleusercontent.com.json"
DROPBOX_ACCESS_TOKEN = "sl.u.AF08HMPZ7V2NEFZbKfoiXRCmq8zQ1CoNTSYScHLm_W4kTmMf4hAGhT9D1I4jP6QwS8P4JTYaN-gZwZmfr9k1k_d67gnN0CCeEbh_HBk3AofquOdGRrvPEvezdni-GdUvI1Rx1LrvR0IzNopcNLFdwSKm1wjzZHPcbAWAMzffW4g5DvJgS70T-RfD1_loMEZfBfzmxHFUen8PE3mBnVr-RCN3ZSWhor49TJtgZBojQq7WDcgitXcdzGALzSjlZanXmDfUK8Y51zpjhKB_DWI4qCx0-PGQbwbWrFOhnJcGDyHljRKzqtkkTN4APZlPxu_5H3UhTrmsk6CpPFFz9dozjuVSPcnuC7dHnSLWPuSmaZyQqlSb3SdgLRkq5S8huUgvqcSCZb2vfwXB7Y-AvHXeP_vZTTuKfiCa6MOKczR3JsJvxQ9RJgKsBpXpJOlasHpRx3Pt2v1o7b12eturrqR108pDk-9n7fE4u3dZCiHxdA5o-LhVslNlnOktfqSWPPSURN3-vpXiyuDK8QjODLqQH5AXNINSS0vrjehYZW4uQjpFJ76F7y9y2rv_JRwOP-kJ8EmKJgEpidWiuhHNOG-qeCSZ9HVRNfnrI9uqJ7My-9J3vVBzL06Z7qh9TGDIabQ2ubepQmJNpc5MaSN-wjIJv-xRk-FEVuw1Jug08QtVk9T7XDWrJ3jV-BK3VVauQl53zxCuUXbBxUAp2k1widx_Y60K9FWHOOFKeJMuVUptgt48nEx_vPyblDzY305i-zNzvTQ4lZcw1xfwMmoshjnghzWRL5uSftIodjdoySouKkyZ0_bQ6LBGVzfaPv-4lSw4bwoRTK6jhYtlLpr-X92co7KY0YOb4plRsFljdmnuc5XwyI0B1QRWJw-CEfgMrCRJ7CJsoW2nSF0a0TkMtVApto8miLqE7xRDWkheYQF6Ftdl6Nr4e7pkwadRXShmSbc_MIMyiTO4eTNTx4R_w43voArWfyoUYb5SZDYI0QN1Q2vu-raFQqL-i8kgTon3UReIZZIenyUcJusGOFN0MKdvYtVJ8TeIzcWj4sIFQtbkTb0fk_d3PIJrz--ep-7oS_UsDmn52l4HQLtp_GWAc0pO_bTvxN-mSZnBS9n-lRXPNKLAUoWTmFVqGXXkDANfn5LLmvwlLXN6SpfVAWAPjd37S4FvSheogiWpTzOKHBUqah32pm8ksLxj9tRDxcwwl5GiKPbSNXMUD6N4X4jmDPY1-HA4q33zUGLgci3JyQUJjqfbCGX_uvp1CUJzIun4Rky49t4wfFrGK29utwVwCMyVIbnF5LAg1q1ED4tPVGH1Gnipzg" # 🔐 PASTE YOUR TOKEN HERE
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
INDEX_FILE_PATH = "dropbox_index.pkl"

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- TEXT EXTRACTION ---
def extract_text_from_file(file_name, file_bytes):
    ext = file_name.lower().split('.')[-1]
    text = ""
    try:
        if ext == 'txt':
            text = file_bytes.decode('utf-8', errors='ignore')
        elif ext == 'docx':
            doc = Document(BytesIO(file_bytes))
            text = '\n'.join([p.text for p in doc.paragraphs])
        elif ext == 'pdf':
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = '\n'.join([page.get_text() for page in doc])
    except Exception as e:
        st.warning(f"Could not read {file_name}: {e}")
    return text

# --- IMAGE ANALYSIS ---
def get_image_labels(image_bytes, file_ext=None):
    try:
        if file_ext in ['avif', 'webp']:
            with Image.open(BytesIO(image_bytes)) as img:
                with BytesIO() as output:
                    img.save(output, format='PNG')
                    image_bytes = output.getvalue()

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.label_detection(image=image)
        labels = [label.description for label in response.label_annotations]
        return ", ".join(labels)
    except Exception as e:
        st.warning(f"Vision API failed: {e}")
        return ""

# --- VIDEO & GIF ANALYSIS ---
def extract_video_frames(file_bytes, file_ext, max_frames=5):
    frames = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if file_ext == 'gif':
            gif = imageio.mimread(tmp_path)
            total_frames = len(gif)
            step = max(1, total_frames // max_frames)
            for i in range(0, total_frames, step):
                img = Image.fromarray(gif[i])
                with BytesIO() as output:
                    img.save(output, format='PNG')
                    frames.append(output.getvalue())
        else:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames)
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    with BytesIO() as output:
                        pil_img.save(output, format='PNG')
                        frames.append(output.getvalue())
            cap.release()
    except Exception as e:
        st.warning(f"Failed to extract frames: {e}")
    return frames

# --- TEXT CHUNKING ---
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# --- INDEXING ---
def create_search_index(dbx):
    st.info("📂 Starting indexing process. This may take a while...")

    try:
        result = dbx.files_list_folder(DROPBOX_FOLDER_PATH, recursive=True)
        files = result.entries
    except Exception as e:
        st.error(f"Dropbox API error: {e}")
        return

    index_data = []
    progress_bar = st.progress(0)
    total_files = len(files)

    for i, file in enumerate(files):
        if isinstance(file, dropbox.files.FileMetadata):
            file_name = file.name
            file_path = file.path_display
            ext = file_name.lower().split('.')[-1]
            progress_bar.progress((i + 1) / total_files, text=f"Processing: {file_name}")

            try:
                _, resp = dbx.files_download(file_path)
                file_bytes = resp.content

                content_to_embed = []

                if ext in ['jpg', 'jpeg', 'png', 'webp', 'avif']:
                    labels = get_image_labels(file_bytes, file_ext=ext)
                    if labels:
                        content_to_embed.append({'type': 'image_labels', 'content': labels})

                elif ext in ['mp4', 'mov', 'avi', 'mkv', 'gif']:
                    frames = extract_video_frames(file_bytes, ext)
                    for frame in frames:
                        labels = get_image_labels(frame)
                        if labels:
                            content_to_embed.append({'type': 'video_frame', 'content': labels})

                elif ext in ['txt', 'docx', 'pdf']:
                    text = extract_text_from_file(file_name, file_bytes)
                    text_chunks = chunk_text(text)
                    for chunk in text_chunks:
                        content_to_embed.append({'type': 'text_chunk', 'content': chunk})

                if content_to_embed:
                    contents = [item['content'] for item in content_to_embed]
                    embeddings = model.encode(contents)

                    try:
                        link = dbx.sharing_create_shared_link_with_settings(file_path).url
                    except dropbox.exceptions.ApiError as e:
                        if e.error.is_shared_link_already_exists():
                            links = dbx.sharing_list_shared_links(file_path).links
                            link = links[0].url if links else None
                        else:
                            link = None

                    for j, item in enumerate(content_to_embed):
                        index_data.append({
                            "file_name": file_name,
                            "link": link,
                            "content": item['content'],
                            "embedding": embeddings[j]
                        })

            except Exception as e:
                st.warning(f"Failed to process {file_name}: {e}")

    with open(INDEX_FILE_PATH, "wb") as f:
        pickle.dump(index_data, f)
    st.success(f"✅ Indexing complete! {len(index_data)} items indexed.")
    return index_data

# --- SEARCH ---
def search_index(query, index):
    if not query or not index:
        return []

    query_embedding = model.encode([query])[0]
    all_embeddings = np.array([item['embedding'] for item in index])
    similarities = util.cos_sim(query_embedding, all_embeddings)[0]

    for i, item in enumerate(index):
        item['similarity'] = similarities[i].item()

    sorted_results = sorted(index, key=lambda x: x['similarity'], reverse=True)
    return [res for res in sorted_results if res['similarity'] > 0.3]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Advanced Dropbox Search", layout="wide")
st.title("File Search (Internal)")

try:
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    dbx.users_get_current_account()
except Exception as e:
    st.error(f"Dropbox authentication failed. Please check your access token. Error: {e}")
    st.stop()

if 'index_data' not in st.session_state:
    if os.path.exists(INDEX_FILE_PATH):
        st.info("Loading existing search index...")
        with open(INDEX_FILE_PATH, "rb") as f:
            st.session_state.index_data = pickle.load(f)
        st.success("Search index loaded.")
    else:
        st.warning("No search index found.")
        st.session_state.index_data = []

if st.button("🔄 Re-build Search Index"):
    st.session_state.index_data = create_search_index(dbx)

if not st.session_state.index_data:
    st.info("The search index is empty. Please build it to enable search.")
else:
    user_input = st.text_input("What are you searching for?", key="search_box")
    if user_input:
        results = search_index(user_input, st.session_state.index_data)

        st.markdown("---")
        st.markdown(f"## ✅ Found {len(results)} relevant results for '{user_input}'")

        if results:
            grouped_results = {}
            for res in results:
                if res['link'] not in grouped_results or res['similarity'] > grouped_results[res['link']]['similarity']:
                    grouped_results[res['link']] = res

            for link, result in list(grouped_results.items())[:10]:
                st.markdown(f"#### 📄 [{result['file_name']}]({result['link']})")
                st.caption(f"Relevance Score: {result['similarity']:.2f}")
                with st.expander("Show most relevant content"):
                    st.info(result['content'])
        else:
            st.info("No matching files found.")
