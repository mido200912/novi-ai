import os, json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# ✅ إنشاء تطبيق FastAPI
app = FastAPI(title="Kepler AI API")

# ✅ تفعيل CORS لتسمح بالاتصال من React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ممكن تخصصها لاحقاً بـ ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ تحميل مفتاح Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 🪐 تحميل بيانات كبلر
with open("kepler_all_exoplanets.json", "r", encoding="utf-8") as f:
    planets = json.load(f)

# 🧠 إعداد نموذج embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

planet_texts = [
    f"{p.get('name', p.get('id'))}: radius {p.get('planet_radius_earth')} Earths, "
    f"temp {p.get('planet_temp_eq_K')} K, star temp {p.get('star_temp_K')} K, "
    f"status {p.get('status')}"
    for p in planets
]

embeddings = embedder.encode(planet_texts, convert_to_tensor=True)

# 🔍 دالة البحث الدلالي في البيانات
def search_local(query, top_k=5):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, embeddings, top_k=top_k)[0]
    return "\n".join([planet_texts[h["corpus_id"]] for h in hits])

# 🪄 إعداد موديل Gemini
model = genai.GenerativeModel("gemini-2.5-flash")

# 🧩 تعريف الـ Schema للطلب
class Question(BaseModel):
    query: str

# ✅ Endpoint رئيسي لاستقبال الأسئلة
@app.post("/ask")
async def ask_gemini(data: Question):
    try:
        query = data.query
        local_context = search_local(query, top_k=8)

        prompt = (
            "أنت خبير فلكي متخصص في الكواكب الخارجية والفضاء. "
            "اعتمد أولًا على بيانات كبلر المرفقة، ثم وسّع إجابتك علميًا.\n\n"
            f"سياق كبلر:\n{local_context}\n\nسؤال المستخدم: {query}"
        )

        response = model.generate_content(prompt)
        return {"answer": response.text.strip()}
    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

# ✅ دعم OPTIONS (عشان React يعمل preflight check)
@app.options("/ask")
async def options_handler():
    return {"ok": True}
