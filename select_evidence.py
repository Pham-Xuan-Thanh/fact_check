import re
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_leading_connectors(text):
    """Xóa bỏ các từ nối gây cụt ý ở đầu câu bằng chứng."""
    connectors = [
        r"^tuy nhiên,?\s*", r"^tuy vậy,?\s*", r"^và,?\s*", r"^nhưng,?\s*",
        r"^mặt khác,?\s*", r"^ngoài ra,?\s*", r"^hơn nữa,?\s*", r"^do đó,?\s*",
        r"^vì vậy,?\s*", r"^trong khi đó,?\s*"
    ]
    cleaned = text.strip()
    if not cleaned: return "" 
    for pattern in connectors:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned[0].upper() + cleaned[1:] if cleaned else ""

def extract_keywords(text):
    """Lấy thực thể số và tên riêng (viết hoa) để ưu tiên thông tin cốt lõi."""
    return set(re.findall(r'\d+(?:\.\d+)?|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', text))

def process_evidence(data, k=2, window_size=2):
    results = []
    tfidf_vectorizer = TfidfVectorizer()

    for entry in data:
        claim = entry['claim']
        claim_keywords = extract_keywords(claim)
        all_candidates = []

        # 1. Trích xuất ứng viên (Sliding Window)
        for doc in entry['documents']:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc['content']) if len(s.strip()) > 5]
            for i in range(len(sentences)):
                group = sentences[i : i + window_size]
                combined_text = " ".join(group)
                if len(combined_text) < 30: continue

                s_keywords = extract_keywords(combined_text)
                matching_keywords = s_keywords.intersection(claim_keywords)
                keyword_score = len(matching_keywords) / len(claim_keywords) if claim_keywords else 0

                all_candidates.append({
                    "text": combined_text,
                    "url": doc.get('url', ''),
                    "doc_score": doc.get('score', 0),
                    "keyword_boost": keyword_score,
                    "date": doc.get('date', '')
                })

        if not all_candidates: continue

        # 2. Xếp hạng bằng BM25
        # Token hóa (tách từ đơn giản bằng khoảng trắng)
        tokenized_corpus = [c['text'].lower().split() for c in all_candidates]
        tokenized_claim = claim.lower().split()
        
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_claim)
        
        # Chuẩn hóa điểm BM25 về [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        rel_scores = np.array([s / max_bm25 for s in bm25_scores])

        # 3. Kết hợp điểm đa tiêu chí
        combined_scores = (
            0.4 * rel_scores +
            0.3 * np.array([c['doc_score'] for c in all_candidates]) +
            0.3 * np.array([c['keyword_boost'] for c in all_candidates])
        )

        # 4. Tối ưu hóa sự đa dạng bằng MMR (Sử dụng Vector để tính độ trùng lặp)
        texts = [c['text'] for c in all_candidates]
        vectors = tfidf_vectorizer.fit_transform(texts)
        
        selected_indices = []
        unselected_indices = list(range(len(all_candidates)))
        
        while len(selected_indices) < k and unselected_indices:
            mmr_scores = []
            for i in unselected_indices:
                relevance = combined_scores[i]
                # Hình phạt trùng lặp (Redundancy)
                redundancy = np.max(cosine_similarity(vectors[i], vectors[selected_indices])) if selected_indices else 0
                
                # MMR Score: Lambda = 0.5
                score = 0.5 * relevance - 0.5 * redundancy
                mmr_scores.append((score, i))
            
            best_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        # 5. Đóng gói kết quả (Chỉ lấy URL, Content, Date)
        final_evidences = []
        for idx in selected_indices:
            cand = all_candidates[idx]
            final_evidences.append({
                "url": cand['url'],
                "content": clean_leading_connectors(cand['text']),
                "date": cand['date']
            })
            
        results.append({"claim": claim, "evidences": final_evidences})
    
    return results

# --- Dữ liệu đầu vào ---
data_to_process = [
    {
        "claim": "Sông Mekong nằm trong top 10 con sống dài nhất thế giới",
        "documents": [
            {
                "score": 0.98,
                "site": "tuoitre.vn",
                "title": "Kỳ vĩ dòng sông Mekong",
                "content": "Sông Mekong dài khoảng 4.350 km, bắt nguồn từ Tây Tạng và chảy qua 6 quốc gia: Trung Quốc, Myanmar, Lào, Thái Lan, Campuchia và Việt Nam. Đây chính là con sông dài nhất khu vực Đông Nam Á.",
                "url": "https://tuoitre.vn/ky-vi-dong-song-mekong-12345.html",
                "date": "2025-10-20"
            },
            {
                "score": 0.75,
                "site": "vi.wikipedia.org",
                "title": "Danh sách sông dài nhất thế giới",
                "content": "Trên thế giới, Sông Mekong dài khoảng 4.350 km, Mekong đứng thứ 12 về độ dài. Tại châu Á, nó chảy qua tiểu vùng Mekong mở rộng bao gồm 6 nước thành viên.",
                "url": "https://vi.wikipedia.org/wiki/Sông_Mê_Kông",
                "date": "2026-01-05"
            },
            {
                "score": 0.40,
                "site": "dulichmientay.com",
                "title": "Du lịch rừng tràm trà sư",
                "content": "Rừng tràm Trà Sư là điểm du lịch nổi tiếng tại An Giang, thuộc vùng đồng bằng sông Cửu Long với hệ sinh thái ngập nước đặc trưng.",
                "url": "https://dulichmientay.com/rung-tram-tra-su",
                "date": "2025-12-12"
            }
        ]
    },
    {
        "claim": "Sản lượng lúa gạo của Việt Nam tập trung chủ yếu ở Đồng bằng sống Cửu Long.",
        "documents": [
            {
                "score": 0.92,
                "site": "vnexpress.net",
                "title": "Vựa lúa miền Tây đứng trước thách thức",
                "content": "Đồng bằng sông Cửu Long (Việt Nam) là vùng hạ lưu cuối cùng của sông Mekong, đóng góp hơn 50% sản lượng lúa gạo và 90% lượng gạo xuất khẩu của cả nước.",
                "url": "https://vnexpress.net/vua-lua-mien-tay-2026.html",
                "date": "2026-02-01"
            },
            {
                "score": 0.88,
                "site": "nongnghiep.vn",
                "title": "Nông nghiệp bền vững vùng Mekong",
                "content": "Các tỉnh như An Giang, Kiên Giang và Đồng Tháp dẫn đầu về diện tích gieo trồng lúa nhờ nguồn nước và phù sa dồi dào từ sông Tiền và sông Hậu.",
                "url": "https://nongnghiep.vn/nong-nghiep-ben-vung",
                "date": "2025-11-30"
            }
        ]
    }
]

# --- Thực thi ---
final_output = process_evidence(data_to_process, k=2, window_size=2)

# Hiển thị kết quả dưới dạng JSON đẹp
print(json.dumps(final_output, indent=4, ensure_ascii=False))