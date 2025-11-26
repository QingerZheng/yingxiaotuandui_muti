"""
Qwen OCR utility class for image content understanding and text recognition.
Provides both image description and text extraction capabilities using Qwen2.5-VL.
"""

import os
from typing import List, Dict, Any, Optional
import dashscope
from dashscope import MultiModalConversation

class QwenOCR:
    """
    A class to handle image content understanding and OCR using Qwen2.5-VL API.
    Provides both image description and text extraction capabilities for RAG systems.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the QwenOCR class.
        
        Args:
            api_key: DashScope API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through DASHSCOPE_API_KEY environment variable")
        
        dashscope.api_key = self.api_key

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image and provide content description using Qwen2.5-VL.
        
        Args:
            image_path: Path to the image file or URL
            
        Returns:
            Combined content including image description and extracted text
        """
        messages = [{
            'role': 'user',
            'content': [
                {
                    'image': image_path
                },
                {
                    'text': (
                        '请先用一句话简要描述这张图片的内容，然后提取并输出图片中的所有文字。'
                        '请严格按照如下格式输出：\n'
                        '描述：<图片内容描述>\n'
                        '文字：<图片中的所有文字>'
                    )
                },
            ]
        }]

        try:
            # Use Qwen2.5-VL model
            response = MultiModalConversation.call(
                model='qwen-vl-max',  # Using the most capable model
                messages=messages
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                # 确保返回字符串类型
                if isinstance(content, list):
                    return '\n'.join(str(item) for item in content)
                elif content is None:
                    return ""
                else:
                    return str(content)
            else:
                raise Exception(f"API call failed with status code {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")

    def analyze_document(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a document image and extract structured information with content description.
        
        Args:
            image_path: Path to the document image or URL
            
        Returns:
            Dictionary containing structured information from the document
        """
        messages = [{
            'role': 'user',
            'content': [
                {
                    'image': image_path
                },
                {
                    'text': (
                        '请先用一句话简要描述这张图片的内容，然后分析此文档并以结构化方式提取关键信息。'
                        '包括所有表格、标题和重要的文本字段。'
                        '请严格按照如下格式输出：\n'
                        '描述：<图片内容描述>\n'
                        '结构化信息：<表格、标题、重要文本等>'
                    )
                },
            ]
        }]

        try:
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages
            )
            
            if response.status_code == 200:
                # Parse the response into structured data
                # This is a simple implementation - you may want to add more sophisticated parsing
                return {
                    'raw_text': response.output.choices[0].message.content,
                    'status': 'success'
                }
            else:
                raise Exception(f"API call failed with status code {response.status_code}")
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def batch_process(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths or URLs
            
        Returns:
            List of results for each image
        """
        results = []
        for image_path in image_paths:
            try:
                text = self.extract_text(image_path)
                results.append({
                    'image_path': image_path,
                    'text': text,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'status': 'error'
                })
        return results
