from typing import List, Dict, Any
import json
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.documents import Document


class ClaimVerifier:
    """
    Verifies claims and predicts labels based on evidence.
    Handles Vietnamese fact-checking with structured evidence format.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["claim", "evidence"],
            template="""Bạn là một chuyên gia kiểm chứng thông tin. Phân tích tuyên bố CHỈ dựa trên các bằng chứng được cung cấp (không dùng kiến thức ngoài, không suy đoán, không bịa).

Tuyên bố: {claim}

Bằng chứng:
{evidence}

Nguyên tắc bắt buộc:
1) Evidence-only: Chỉ dùng thông tin có trong Bằng chứng. Nếu thiếu dữ kiện then chốt để kết luận đúng/sai → NOT ENOUGH INFO.
2) Nhạy với sai lệch nhỏ: Nếu tuyên bố sai ở bất kỳ chi tiết QUAN TRỌNG nào (con số, ngày/tháng/năm, chức vị, tên người/tổ chức, địa điểm, phạm vi “tất cả/luôn/duy nhất”, quan hệ nhân quả...) và bằng chứng cho thấy rõ → REFUTED.
3) SUPPORTED chỉ khi bằng chứng xác nhận TRỰC TIẾP nội dung tuyên bố và không có mâu thuẫn trọng yếu.
4) Xử lý mâu thuẫn: Nếu các bằng chứng mâu thuẫn nhau, ưu tiên nguồn uy tín hơn trong phạm vi evidence; nếu vẫn không thể kết luận chắc → NOT ENOUGH INFO và nêu rõ mâu thuẫn.
5) Không “hợp thức hóa” bằng chứng yếu: Nếu bằng chứng chỉ nói chung chung/gián tiếp/ý kiến/không đề cập trực tiếp claim thì không đủ để SUPPORTED.

Dựa trên bằng chứng, phân loại tuyên bố thành:
- SUPPORTED (Được hỗ trợ): Bằng chứng xác nhận tuyên bố là đúng
- REFUTED (Bị bác bỏ): Bằng chứng mâu thuẫn với tuyên bố hoặc cho thấy chi tiết quan trọng là sai
- NOT ENOUGH INFO (Không đủ thông tin): Bằng chứng không đủ/không liên quan trực tiếp để xác minh tuyên bố

Tập trung vào các nguồn dữ liệu uy tín (báo, trang chính phủ, tài liệu chính thức) và đặc biệt chú ý các thông tin sai lệch nhỏ (con số, chức vị, mốc thời gian...).

Trả lời CHÍNH XÁC theo định dạng JSON sau (không thêm markdown backticks, không thêm bất kỳ nội dung nào khác ngoài JSON):
{{
    "label": "SUPPORTED hoặc REFUTED hoặc NOT ENOUGH INFO",
    "confidence": 0.85,
    "reasoning": "Giải thích ngắn gọn bằng tiếng Việt về lý do phân loại, nêu rõ chi tiết nào trong claim được hỗ trợ/bị mâu thuẫn/thiếu thông tin",
    "evidence_used": [
        {{
            "evidence_id": "doc_1_wikipedia",
            "snippet": "Trích dẫn ngắn từ bằng chứng liên quan nhất (ưu tiên nguyên văn)",
            "relevance": "Tại sao đoạn này hỗ trợ/bác bỏ/cho thấy không đủ thông tin"
        }}
    ]
}}

"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def format_evidence(self, evidences: List[Dict[str, Any]]) -> str:
        """
        Format evidence list as structured text.

        Args:
            evidences: List of evidence dicts with format:
                {
                    "evidence_id": "doc_1_wikipedia",
                    "site": "https://...",
                    "text": "Evidence content..."
                }

        Returns:
            Formatted evidence string
        """
        if not evidences:
            return "Không có bằng chứng nào được tìm thấy."

        evidence_parts = []
        for evidence in evidences:
            evidence_id = evidence.get('evidence_id', 'unknown')
            site = evidence.get('site', 'Unknown source')
            text = evidence.get('text', '')

            evidence_parts.append(
                f"[{evidence_id}]\n"
                f"Nguồn: {site}\n"
                f"Nội dung: {text}\n"
                f"{'-'*50}"
            )

        return "\n\n".join(evidence_parts)

    def verify_claim(self, claim: str, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify a single claim against evidence.

        Args:
            claim: The claim text
            evidences: List of evidence dictionaries with structure:
                {
                    "evidence_id": "doc_1_wikipedia",
                    "site": "https://...",
                    "text": "Evidence content..."
                }

        Returns:
            Verification result with label, confidence, reasoning, and evidence references
        """
        try:
            # Format evidence
            formatted_evidence = self.format_evidence(evidences)

            # Run verification chain
            response = self.chain.run(claim=claim, evidence=formatted_evidence)

            # Clean response (remove markdown backticks if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                # Remove markdown code blocks
                cleaned_response = cleaned_response.split('```')[1]
                if cleaned_response.startswith('json'):
                    cleaned_response = cleaned_response[4:]
            cleaned_response = cleaned_response.strip()

            # Parse JSON response
            result = json.loads(cleaned_response)

            # Add original evidence references to result
            result['evidences'] = evidences

            # Ensure label is standardized
            label_mapping = {
                'SUPPORTED': 'SUPPORTED',
                'REFUTED': 'REFUTED',
                'NOT ENOUGH INFO': 'NOT ENOUGH INFO',
                'KHÔNG ĐỦ THÔNG TIN': 'NOT ENOUGH INFO',
                'ĐƯỢC HỖ TRỢ': 'SUPPORTED',
                'BỊ BÁC BỎ': 'REFUTED'
            }
            result['label'] = label_mapping.get(result['label'].upper(), result['label'])

            # Add metadata
            result['claim'] = claim
            result['num_evidences_used'] = len(evidences)

            return result

        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return {
                'claim': claim,
                'label': 'NOT ENOUGH INFO',
                'confidence': 0.0,
                'reasoning': f'Lỗi phân tích kết quả: {str(e)}. Response: {response[:200]}',
                'evidence_used': [],
                'evidences': evidences,
                'num_evidences_used': len(evidences),
                'error': str(e)
            }
        except Exception as e:
            # General error handling
            return {
                'claim': claim,
                'label': 'NOT ENOUGH INFO',
                'confidence': 0.0,
                'reasoning': f'Lỗi trong quá trình xác minh: {str(e)}',
                'evidence_used': [],
                'evidences': evidences,
                'num_evidences_used': len(evidences),
                'error': str(e)
            }

    def verify_claim_from_dict(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a claim from dictionary format.

        Args:
            claim_data: Dictionary with format:
                {
                    "claim": "Claim text...",
                    "evidences": [
                        {
                            "evidence_id": "doc_1_wikipedia",
                            "site": "https://...",
                            "text": "Evidence content..."
                        }
                    ]
                }

        Returns:
            Verification result
        """
        claim = claim_data.get('claim', '')
        evidences = claim_data.get('evidences', [])

        return self.verify_claim(claim, evidences)

    def verify_all_claims(self,
                          claims_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify multiple claims from list of dictionaries.

        Args:
            claims_data: List of claim dictionaries with format:
                [
                    {
                        "claim": "Claim text...",
                        "evidences": [
                            {
                                "evidence_id": "doc_1_wikipedia",
                                "site": "https://...",
                                "text": "Evidence content..."
                            }
                        ]
                    }
                ]

        Returns:
            List of verification results
        """
        results = []

        for idx, claim_data in enumerate(claims_data, 1):
            claim = claim_data.get('claim', '')
            evidences = claim_data.get('evidences', [])

            print(f"Verifying claim {idx}/{len(claims_data)}: {claim[:50]}...")

            # Verify the claim
            result = self.verify_claim(claim, evidences)
            result['claim_id'] = idx

            results.append(result)

        return results