# modules/llm_modules.py
"""
LLM (Large Language Model) 엔진 통합 모듈
각 LLM을 클래스로 관리하여 독립적 사용 가능
"""
import json
import re
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from google import genai
from configs import OPENAI_API_KEY, GEMINI_API_KEY, ACTION_JSON_PATH, OUTPUT_DIR
from configs.prompt import ROBOT_COMMAND_PROMPT as PROMPT_TEMPLATE
from utils.logger import Logger


# assets 폴더 자동 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# JSON 추출 및 저장 유틸리티
# ==========================================
def extract_json(text: Optional[str]) -> Dict[str, Any]:
    """
    LLM 응답에서 JSON 추출
    Args:
        text: LLM 응답 텍스트
    Returns:
        파싱된 JSON 딕셔너리
    """
    if not text:
        Logger.warning("LLM 응답이 비어있습니다")
        return {"action": None, "aux0": None, "aux1": "응답 텍스트가 없습니다."}
    
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0).strip())
        
        Logger.warning("JSON 형식을 찾을 수 없습니다")
        return {"action": None, "aux0": None, "aux1": text.strip()}
    except json.JSONDecodeError as e:
        Logger.error(f"JSON 파싱 실패: {e}")
        return {"action": None, "aux0": None, "aux1": text.strip()}
    except Exception as e:
        Logger.error(f"예상치 못한 오류: {e}")
        return {"action": None, "aux0": None, "aux1": text.strip()}


def save_action_json(command_dict: Dict[str, Any], filepath: str = None) -> None:
    """
    LLM 결과를 action.json 파일로 저장
    Args:
        command_dict: 명령 딕셔너리
        filepath: 저장 경로 (기본값: ACTION_JSON_PATH)
    """
    filepath = filepath or ACTION_JSON_PATH
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(command_dict, f, ensure_ascii=False, indent=2)
        Logger.debug(f"action.json 저장 완료: {filepath}")
    except Exception as e:
        Logger.error(f"action.json 저장 실패: {e}")


# ==========================================
# 1. GPT-4o (OpenAI)
# ==========================================
class GPT4oLLM:
    """OpenAI GPT-4o API를 사용한 LLM"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    async def ask(self, text: str) -> Dict[str, Any]:
        """
        사용자 입력을 LLM에 전달하고 명령 JSON 반환
        Args:
            text: 사용자 음성/텍스트 입력
        Returns:
            {"action": str, "aux0": str, "aux1": str}
        """
        instructions = PROMPT_TEMPLATE.replace("{stt_text}", text)
        
        try:
            Logger.debug("GPT-4o 요청 시작")
            
            # 비동기 환경에서 동기 라이브러리 안전 실행
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a robot control assistant. Respond only in JSON format."},
                        {"role": "user", "content": instructions}
                    ],
                    response_format={"type": "json_object"}
                )
            )
            
            res_text = response.choices[0].message.content
            Logger.debug(f"GPT-4o 응답 (raw): {res_text}")
            
            # 딕셔너리로 파싱
            parsed_dict = extract_json(res_text)
            
            # action.json 저장
            save_action_json(parsed_dict)
            
            # globals에 raw 문자열도 저장 (원본 코드 호환용)
            from configs import globals as g
            g.set_voice_command(parsed_dict, raw_json=res_text)
            
            return parsed_dict
        
        except Exception as e:
            Logger.error(f"GPT 통신 오류: {e}")
            error_dict = {"action": None, "aux0": None, "aux1": f"GPT 통신 에러: {str(e)}"}
            save_action_json(error_dict)
            from configs import globals as g
            g.set_voice_command(error_dict, raw_json=json.dumps(error_dict))
            return error_dict


# ==========================================
# 2. Gemini 2.0 (Google)
# ==========================================
class GeminiLLM:
    """Google Gemini 2.0 API를 사용한 LLM"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
    
    async def ask(self, text: str) -> Dict[str, Any]:
        """
        사용자 입력을 LLM에 전달하고 명령 JSON 반환
        Args:
            text: 사용자 음성/텍스트 입력
        Returns:
            {"action": str, "aux0": str, "aux1": str}
        """
        try:
            Logger.debug("Gemini 2.0 요청 시작")
            
            prompt = PROMPT_TEMPLATE.replace("{stt_text}", text)
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            res_text = response.text
            Logger.debug(f"Gemini 응답 (raw): {res_text}")
            
            # 딕셔너리로 파싱
            parsed_dict = extract_json(res_text)
            
            # action.json 저장 (백업/로깅용)
            save_action_json(parsed_dict)
            
            # globals에 raw 문자열도 저장 (원본 코드 호환용)
            from configs import globals as g
            g.set_voice_command(parsed_dict, raw_json=res_text)
            
            return parsed_dict
        
        except Exception as e:
            Logger.error(f"Gemini 통신 오류: {e}")
            error_dict = {"action": None, "aux0": None, "aux1": f"Gemini 에러: {str(e)}"}
            save_action_json(error_dict)
            from configs import globals as g
            g.set_voice_command(error_dict, raw_json=json.dumps(error_dict))
            return error_dict
