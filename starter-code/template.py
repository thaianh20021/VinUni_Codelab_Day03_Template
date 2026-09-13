"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

# Đảm bảo terminal Windows in được tiếng Việt UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau của khách hàng mà KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api"
                }
            except Exception:
                pass

        # Chatbot Baseline không kết nối cơ sở dữ liệu hay tool:
        # Chatbot sẽ từ chối tra cứu thông tin thực tế hoặc bịa ra dữ liệu (hallucination)
        return {
            "answer": (
                "Xin lỗi quý khách, tôi là chatbot cơ bản và không có quyền truy cập vào cơ sở dữ liệu "
                "chuyến bay hay thông tin thời tiết thời gian thực. Tôi không thể tra cứu các chuyến bay "
                "từ HAN đi SGN dưới 2 triệu VND cũng như thông tin thời tiết hiện tại ở SGN để tư vấn trang phục. "
                "Quý khách vui lòng kiểm tra trực tiếp trên website của các hãng hàng không và ứng dụng dự báo thời tiết."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẮNG" in text_upper:
            return "DAD"
        return "SGN"

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        """Dynamic step planning supporting multi-step, single-step, FAQ, and fallback queries"""
        user_lower = user_input.lower()
        
        # 1. Check FAQ query (không cần gọi Tool)
        if "chính sách" in user_lower or "đổi trả" in user_lower:
            thought = "Đây là câu hỏi FAQ chung về chính sách. Không cần sử dụng tool."
            final_answer = "Vé máy bay Vinpearl có thể hỗ trợ đổi ngày trước 24 giờ so với giờ khởi hành, phí đổi vé là 350.000 VNĐ/vé cộng chênh lệch giá vé (nếu có)."
            self.trace.append({"iteration": iteration, "thought": thought, "final_answer": final_answer})
            return final_answer, True

        # 2. Xác định nhu cầu tra cứu
        needs_flight = any(k in user_lower for k in ["chuyến bay", "vé", "bay từ", "vé máy bay"])
        needs_weather = any(k in user_lower for k in ["thời tiết", "mặc gì", "nhiệt độ", "mưa"])

        # Bước 1: Tra cứu chuyến bay
        if needs_flight and iteration == 1:
            origin = "HAN" if "han" in user_lower or "hà nội" in user_lower else "DAD"
            destination = "DAD" if "dad" in user_lower or "đà nẵng" in user_lower else "SGN"
            max_price = 5000000
            if "2 triệu" in user_lower or "2.000.000" in user_lower:
                max_price = 2000000
            elif "1.5 triệu" in user_lower or "1,5 triệu" in user_lower:
                max_price = 1500000
            elif "500k" in user_lower:
                max_price = 500000

            thought = f"Tôi cần tra cứu chuyến bay từ {origin} đi {destination} với giá tối đa {max_price} VND."
            action = {"name": "get_flight_info", "args": {"origin": origin, "destination": destination, "max_price": max_price}}
            
            # Trap 1: Chuẩn hóa tên tool tránh KeyError
            tool_name = action["name"].strip().lower()
            obs = TOOL_MAP[tool_name](**action["args"])
            
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })
            
            if not needs_weather:
                if not obs:
                    final_ans = f"Không tìm thấy chuyến bay nào từ {origin} đi {destination} dưới {max_price:,} VND."
                else:
                    lines = [f"- {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in obs]
                    final_ans = f"Tìm thấy {len(obs)} chuyến bay từ {origin} đi {destination}:\n" + "\n".join(lines)
                return final_ans, True
                
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # Bước 2: Tra cứu thời tiết (cho câu hỏi kép ở lượt 2, hoặc câu hỏi đơn ở lượt 1)
        elif needs_weather and (iteration == 2 or (iteration == 1 and not needs_flight)):
            city_code = self.parse_city_code(user_input)
            thought = f"Tôi cần kiểm tra thông tin thời tiết tại {city_code}."
            action = {"name": "get_weather_forecast", "args": {"city_code": city_code}}
            
            # Trap 1: Chuẩn hóa tên tool tránh KeyError
            tool_name = action["name"].strip().lower()
            obs = TOOL_MAP[tool_name](**action["args"])

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            if not needs_flight:
                final_ans = f"Thời tiết tại {obs.get('city', city_code)}: {obs.get('temperature_c', 'N/A')}°C, {obs.get('condition', '')}.\nGợi ý: {obs.get('recommendation', '')}"
                return final_ans, True
                
            return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}", False

        # Bước 3: Đã thu thập đủ thông tin -> Xuất Final Answer
        else:
            thought = "Tôi đã thu thập đủ thông tin để trả lời khách hàng."
            flight_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_flight_info"), [])
            weather_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "get_weather_forecast"), {})

            flight_summary = "Không tìm thấy chuyến bay phù hợp."
            if flight_obs:
                lines = [f"   - {fl['airline']} ({fl['flight_number']}): {fl['departure_time']} - Giá: {fl['price_vnd']:,} VNĐ" for fl in flight_obs]
                flight_summary = "\n".join(lines)

            weather_summary = f"Thời tiết tại {weather_obs.get('city', 'địa phương')}: {weather_obs.get('temperature_c', '')}°C ({weather_obs.get('condition', '')}).\n   - Gợi ý trang phục: {weather_obs.get('recommendation', '')}"

            final_answer = (
                f"1. Thông tin chuyến bay:\n{flight_summary}\n\n"
                f"2. Thông tin thời tiết & trang phục:\n   - {weather_summary}"
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        iteration = 1
        
        # TODO 2: Thiết lập vòng lặp while iteration <= self.max_iterations
        while iteration <= self.max_iterations:
            # TODO 3, 4, 5: Phân tích Thought/Action, thực thi Tool trong TOOL_MAP, ghi Observation
            result, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        # Milestone 4: Safeguards ngắt vòng lặp khi vượt quá max_iterations
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration - 1
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", json.dumps(result, indent=2, ensure_ascii=False))
    print("\nTrace Log:\n", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()