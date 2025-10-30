from sentence_transformers import SentenceTransformer, util
from PIL import Image
from io import BytesIO
import re
import requests
import torch


# Now we load and encode the images
def load_image(url_or_path):
    """Load an image from a URL or local path.

    For URLs we:
    - send a browser-like User-Agent
    - follow redirects
    - check HTTP status and content-type
    - read content into BytesIO and let PIL open it

    Raises a ValueError with a helpful message when the image can't be loaded.
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        headers = {
            "User-Agent": "curl/8.14.1",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        try:
            print("Going to fetch URL:", url_or_path)
            resp = requests.get(
                url_or_path,
                stream=True,
                allow_redirects=True,
                timeout=15,
                headers=headers,
            )
            print(f"Response status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Response headers: {dict(resp.headers)}")
        except Exception as e:
            raise ValueError(f"Failed to fetch URL {url_or_path}: {e}")

        if resp.status_code != 200:
            # Special-case Unsplash download endpoints which may return 403 or require different handling
            if "unsplash.com" in url_or_path:
                try:
                    # Try to follow redirects manually and get the actual image URL
                    # First, try without redirects to see if we get a redirect response
                    resp_no_redirect = requests.get(
                        url_or_path, allow_redirects=False, timeout=15, headers=headers
                    )

                    if resp_no_redirect.status_code in [301, 302, 303, 307, 308]:
                        # Follow the redirect to get the actual image
                        redirect_url = resp_no_redirect.headers.get("Location")
                        if redirect_url:
                            print(f"Following redirect to: {redirect_url}")
                            resp = requests.get(
                                redirect_url,
                                stream=True,
                                allow_redirects=True,
                                timeout=15,
                                headers=headers,
                            )
                            if resp.status_code == 200:
                                print(
                                    f"Successfully got image from redirect: {resp.status_code}"
                                )
                            else:
                                print(
                                    f"Redirect failed with status: {resp.status_code}"
                                )

                    # If still not successful, try the fallback method (og:image extraction)
                    if resp.status_code != 200:
                        # Derive photo page URL by removing '/download' and query params
                        page_url = url_or_path.split("/download")[0]
                        page_headers = {"User-Agent": headers["User-Agent"]}
                        page_resp = requests.get(
                            page_url, headers=page_headers, timeout=15
                        )
                        if page_resp.status_code == 200 and page_resp.text:
                            m = re.search(
                                r'<meta property="og:image" content="([^"]+)"',
                                page_resp.text,
                            )
                            if m:
                                img_url = m.group(1)
                                print(f"Found og:image URL: {img_url}")
                                resp = requests.get(
                                    img_url,
                                    stream=True,
                                    allow_redirects=True,
                                    timeout=15,
                                    headers=headers,
                                )
                            else:
                                raise ValueError(
                                    f"Could not find og:image on Unsplash page {page_url}"
                                )
                        else:
                            raise ValueError(
                                f"Failed to fetch Unsplash page {page_url}: status {page_resp.status_code}"
                            )
                except Exception as e:
                    raise ValueError(
                        f"Unexpected status code {resp.status_code} when fetching {url_or_path} (and failed to recover): {e}"
                    )
            else:
                raise ValueError(
                    f"Unexpected status code {resp.status_code} when fetching {url_or_path}"
                )

        content_type = resp.headers.get("content-type", "")
        # Some download endpoints (like unsplash) may redirect through HTML; still try to open bytes
        try:
            data = resp.content
        except Exception as e:
            raise ValueError(f"Failed to read content from {url_or_path}: {e}")

        # Quick sanity: if content-type looks like HTML, warn the user but still attempt to open
        if "html" in content_type.lower():
            raise ValueError(
                f"URL {url_or_path} returned HTML (content-type={content_type}). This is not an image. Try using the direct image URL."
            )

        try:
            return Image.open(BytesIO(data))
        except Exception:
            # last attempt: try opening from raw bytes without relying on content-type
            try:
                return Image.open(BytesIO(data))
            except Exception as e:
                raise ValueError(
                    f"PIL failed to identify image from {url_or_path}: {e}"
                )
    else:
        try:
            return Image.open(url_or_path)
        except Exception as e:
            raise ValueError(f"PIL failed to open local image {url_or_path}: {e}")


# We load 3 images. We now use direct image URLs instead of Unsplash download URLs
# to avoid the 403 error issue
img_paths = [
    # Dog image - direct URL from Unsplash
    "https://images.unsplash.com/photo-1547494912-c69d3ad40e7f?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&w=640",
    # Cat image - direct URL from Unsplash
    "https://images.unsplash.com/photo-1511044568932-338cba0ad803?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&w=640",
    # Beach image - direct URL from Unsplash
    "https://images.unsplash.com/photo-1506953823976-52e1fdc0149a?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&w=640",
]

images = [load_image(img) for img in img_paths]

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer("clip-ViT-B-32")

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")


# Map images to the vector space. Try passing PIL images first (commonly supported).
try:
    img_embeddings = img_model.encode(images)
except TypeError:
    # Fallback: convert to RGB numpy arrays and retry
    try:
        import numpy as _np

        def pil_to_rgb_array(img: Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            return _np.array(img)

        img_arrays = [pil_to_rgb_array(im) for im in images]
        img_embeddings = img_model.encode(img_arrays)
    except Exception as e:
        raise RuntimeError(f"Failed to encode images with the image model: {e}")

# Now we encode our text:
texts = [
    "A dog in the snow",
    "Eine Katze",  # German: A cat
    "Una playa con palmeras.",  # Spanish: a beach with palm trees
]

text_embeddings = text_model.encode(texts)

# Compute cosine similarities:
cos_sim = util.cos_sim(text_embeddings, img_embeddings)

for text, scores in zip(texts, cos_sim):
    max_img_idx = torch.argmax(scores)
    print("Text:", text)
    print("Score:", scores[max_img_idx])
    print("Path:", img_paths[max_img_idx], "\n")
