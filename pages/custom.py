import numpy as np
import pandas as pd
import streamlit as st

st.subheader("📉 Gradient Descent Optimizer Studio")
st.write(
    "จำลองการทำงานของ **Gradient Descent** หัวใจหลักในการเทรนโมเดล AI "
    "เพื่อหาค่า $x$ ที่ทำให้ฟังก์ชันความผิดพลาด (Loss Function) มีค่าน้อยที่สุด"
)
st.latex(r"x_{\text{new}} = x_{\text{old}} - \alpha \cdot f'(x_{\text{old}})")
st.markdown("---")

# --- 1. INPUT SECTION ---
st.markdown("### 1. กำหนดฟังก์ชันและพารามิเตอร์ (Input)")

func_choice = st.selectbox(
    "เลือก Loss Function ที่ต้องการทดสอบ:",
    [
        "1. Convex Bowl (พาราโบลาพื้นฐาน): f(x) = x² - 4x + 5",
        "2. Non-Convex (มีหลุมหลอก Local Minimum): f(x) = 0.1x⁴ - 0.8x² + 0.3x + 3",
        "3. Custom Quadratic (กำหนดสมการเอง): f(x) = ax² + bx + c",
    ],
)

# กำหนดสมการ f(x) และอนุพันธ์ f'(x)
if func_choice.startswith("1."):
    a, b, c = 1.0, -4.0, 5.0
    f = lambda x: a * x**2 + b * x + c
    df = lambda x: 2 * a * x + b
    formula_text = r"f(x) = x^2 - 4x + 5 \quad \Rightarrow \quad f'(x) = 2x - 4"
    x_min_plot, x_max_plot = -3.0, 7.0
    default_x0 = 6.0
elif func_choice.startswith("2."):
    f = lambda x: 0.1 * x**4 - 0.8 * x**2 + 0.3 * x + 3.0
    df = lambda x: 0.4 * x**3 - 1.6 * x + 0.3
    formula_text = r"f(x) = 0.1x^4 - 0.8x^2 + 0.3x + 3 \quad \Rightarrow \quad f'(x) = 0.4x^3 - 1.6x + 0.3"
    x_min_plot, x_max_plot = -3.5, 3.5
    default_x0 = 3.0
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.number_input("ค่า a (ควร > 0):", value=1.0, step=0.5)
    with c2:
        b = st.number_input("ค่า b:", value=-2.0, step=0.5)
    with c3:
        c = st.number_input("ค่า c:", value=2.0, step=0.5)
    f = lambda x: a * x**2 + b * x + c
    df = lambda x: 2 * a * x + b
    formula_text = rf"f(x) = {a}x^2 + ({b})x + ({c}) \quad \Rightarrow \quad f'(x) = {2*a}x + ({b})"
    x_min_plot, x_max_plot = -6.0, 6.0
    default_x0 = 4.0

st.latex(formula_text)

col1, col2, col3 = st.columns(3)
with col1:
    x0 = st.slider(
        "จุดเริ่มต้น (Initial x₀):",
        min_value=float(x_min_plot),
        max_value=float(x_max_plot),
        value=float(default_x0),
        step=0.1,
    )
with col2:
    learning_rate = st.slider(
        "Learning Rate (α):",
        min_value=0.01,
        max_value=1.20,
        value=0.15,
        step=0.01,
        help="ลองปรับสูงๆ (เช่น > 1.0) เพื่อดูอาการค่าพุ่งหนี (Diverge)",
    )
with col3:
    max_iters = st.slider(
        "จำนวนรอบสูงสุด (Iterations):",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
    )

# --- 2. PROCESSING (GRADIENT DESCENT LOOP) ---
tolerance = 1e-4
x_curr = float(x0)
history = []
status = "max_iter"

for step in range(max_iters + 1):
    loss = float(f(x_curr))
    grad = float(df(x_curr))

    history.append(
        {
            "Step": step,
            "x": x_curr,
            "Loss f(x)": loss,
            "Gradient f'(x)": grad,
        }
    )

    # ตรวจจับอาการค่าพุ่งหนี (Divergence)
    if abs(x_curr) > 1e4 or not np.isfinite(loss):
        status = "diverged"
        break

    # ตรวจจับการลู่เข้าจุดต่ำสุด (Convergence)
    if abs(grad) < tolerance:
        status = "converged"
        break

    # อัปเดตค่า x ด้วยสูตร Gradient Descent
    x_curr = x_curr - learning_rate * grad

df_history = pd.DataFrame(history)
final_row = df_history.iloc[-1]

st.markdown("---")

# --- 3. OUTPUT & VISUALIZATION ---
st.markdown("### 2. ผลการคำนวณและกราฟจำลอง (Output)")

# แสดงสถานะการเทรน
if status == "converged":
    st.success(
        f"✅ **Converged!** พบจุดต่ำสุดที่ $x = {final_row['x']:.4f}$ "
        f"(Loss = {final_row['Loss f(x)']:.4f}) โดยใช้ไป {int(final_row['Step'])} รอบ"
    )
elif status == "diverged":
    st.error(
        "❌ **Diverged!** ค่าพุ่งออกนอกขอบเขตเนื่องจาก **Learning Rate (α) สูงเกินไป** "
        "ลองลดค่า α ลงแล้วสังเกตความเปลี่ยนแปลง"
    )
else:
    st.warning(
        f"⚠️ **ยังไม่ถึงจุดต่ำสุดสนิท** (ครบ {max_iters} รอบแล้ว, ความชันเหลือ {final_row['Gradient f\'(x)']:.4f}) "
        "ลองเพิ่มจำนวนรอบ หรือปรับ Learning Rate (α) ให้เหมาะสม"
    )

# แสดงตัวเลขสำคัญ 4 ช่อง
m1, m2, m3, m4 = st.columns(4)
m1.metric("จุดเริ่มต้น (x₀)", f"{x0:.2f}")
m2.metric("จุดสุดท้าย (Final x)", f"{final_row['x']:.4f}")
m3.metric("ค่า Loss ต่ำสุด", f"{final_row['Loss f(x)']:.4f}")
m4.metric("จำนวนรอบที่ใช้", f"{int(final_row['Step'])} / {max_iters}")

# กราฟที่ 1: เส้นโค้ง Loss Function + เส้นทางจุดที่ไหลลง
st.markdown("#### 📈 เส้นทางการไหลลงสู่จุดต่ำสุด (Loss Landscape)")
x_curve = np.linspace(x_min_plot, x_max_plot, 200)
curve_data = [{"x": float(xv), "y": float(f(xv))} for xv in x_curve]

# กรองเฉพาะจุดที่อยู่ในขอบเขตกราฟเพื่อไม่ให้กราฟเพี้ยนตอน Diverge
path_data = [
    {
        "x": float(r["x"]),
        "y": float(r["Loss f(x)"]),
        "Step": int(r["Step"]),
    }
    for _, r in df_history.iterrows()
    if x_min_plot - 2 <= r["x"] <= x_max_plot + 2 and np.isfinite(r["Loss f(x)"])
]

vega_spec = {
    "height": 340,
    "layer": [
        {
            "data": {"values": curve_data},
            "mark": {"type": "line", "color": "#4A90E2", "strokeWidth": 2.5},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": "Parameter (x)"},
                "y": {"field": "y", "type": "quantitative", "title": "Loss f(x)"},
            },
        },
        {
            "data": {"values": path_data},
            "mark": {"type": "line", "color": "#FF4B4B", "strokeDash": [4, 4]},
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
                "order": {"field": "Step", "type": "quantitative"},
            },
        },
        {
            "data": {"values": path_data},
            "mark": {"type": "circle", "size": 80, "color": "#FF4B4B"},
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
                "tooltip": [
                    {"field": "Step", "type": "quantitative"},
                    {"field": "x", "type": "quantitative", "format": ".4f"},
                    {"field": "y", "type": "quantitative", "title": "Loss", "format": ".4f"},
                ],
            },
        },
    ],
}
st.vega_lite_chart(vega_spec, use_container_width=True)

# กราฟที่ 2: ค่า Loss ที่ลดลงในแต่ละรอบ
if status != "diverged":
    st.markdown("#### 📉 แนวโน้มค่า Loss ในแต่ละรอบ (Loss vs. Iteration)")
    st.line_chart(df_history.set_index("Step")[["Loss f(x)"]])

# ตารางแสดงรายละเอียดทีละรอบ
with st.expander("📋 ดูตารางคำนวณทีละรอบ (Step-by-Step Table)"):
    st.dataframe(df_history, use_container_width=True, hide_index=True)

# สรุปการประยุกต์ใช้ใน AI (ช่วยตอนพรีเซนต์)
st.info(
    "💡 **การประยุกต์ใช้ใน AI:** อัลกอริทึม Gradient Descent ใช้ในการปรับค่าน้ำหนัก (Weights) "
    "ของ Neural Networks และ Machine Learning เพื่อลดค่าความผิดพลาด (Loss) ให้น้อยที่สุด "
    "หากตั้ง Learning Rate ต่ำเกินไปโมเดลจะเรียนรู้ช้า แต่หากสูงเกินไปค่าจะแกว่งจนหาคำตอบไม่ได้ (Diverge)"
)