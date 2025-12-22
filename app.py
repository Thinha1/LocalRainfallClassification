from flask import Flask, request, render_template, send_from_directory
import io
import torch
from torchvision import transforms
from flask_cors import CORS
import os, base64
from predict import load_model, predict_image
import tifffile as tiff
from PIL import Image
from lib import *

device = 'cpu'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app)
#Cấu hình email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # Hoặc 465 nếu dùng SSL/TLS
SENDER_EMAIL = "Mail_cua_ban"  # Email gửi
SENDER_PASSWORD = "mat_khau_cua_ban"  # Mật khẩu ứng dụng (App Password)
RECEIVER_EMAIL = "Mail_ban_muon_nhan" # Email nhận cảnh báo

# Đọc CSV
df = pd.read_csv("rainfall_data.csv")

df['date'] = pd.to_datetime(df['date'], format='mixed')

df['year'] = df['date'].dt.year

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.6),
    nn.Linear(num_features, 3) 
)

checkpoint_path = "model/model_076.pth"
model = load_model(checkpoint_path, device='cpu')

class_names = ['not_rain', 'medium_rain', 'heavy_rain']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def send_alert_email(subject, pred_label, risk_level, pred_prob, img_data_b64, receiver_email):
    try:
        # 1. Khởi tạo email
        msg = MIMEMultipart('related') 
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = subject
        
        # 2. Định nghĩa HTML Body
        html_body = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }}
                    .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; }}
                    .header {{ background-color: #dc3545; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .result {{ font-size: 1.2em; font-weight: bold; color: #dc3545; margin-top: 10px; }}
                    .image-container {{ text-align: center; margin: 20px 0; border-top: 1px solid #eee; padding-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2>⚠️ CẢNH BÁO MƯA LỚN CẤP ĐỘ CAO ⚠️</h2>
                    </div>
                    <div class="content">
                        <p>Hệ thống dự báo thời tiết vệ tinh đã phát hiện một sự kiện nguy hiểm tiềm tàng.</p>
                        <p><strong>ĐÃ GỬI TỚI:</strong> {receiver_email}</p>
                        <hr>
                        <p><strong>Dự đoán Phân loại:</strong> <span class="result">{pred_label}</span></p>
                        <p><strong>Mức độ Nguy cơ:</strong> <span class="result">{risk_level}</span></p>
                        <p><strong>Độ tin cậy:</strong> {pred_prob*100:.2f}%</p>
                        
                        <div class="image-container">
                            <h3>Ảnh Vệ tinh (Đã xử lý):</h3>
                            <img src="cid:satellite_image" alt="Ảnh vệ tinh cảnh báo" style="max-width: 100%; height: auto; border: 2px solid #dc3545; border-radius: 4px;">
                            <p style="font-size: 0.8em; color: #888;">Nguồn: Ảnh MODIS Aqua</p>
                        </div>
                        
                        <p style="margin-top: 30px;">Vui lòng theo dõi tình hình và kiểm tra ứng dụng web để có dữ liệu cập nhật.</p>
                        <p style="text-align: center; font-size: 0.7em; color: #888; margin-top: 40px;">Hệ thống Dự báo Thời tiết Vệ tinh (Deep Learning Project)</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))

        # 3. Nhúng ảnh (Inline Image)
        if img_data_b64:
            image_data = base64.b64decode(img_data_b64)
            image = MIMEImage(image_data, name='satellite_image.png')
            image.add_header('Content-ID', '<satellite_image>') 
            image.add_header('Content-Disposition', 'inline', filename='satellite_image.png')
            msg.attach(image)

        # 4. Kết nối và gửi email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Lỗi gửi email: {e}")
        return False

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def predict():
    label_final = ""
    img_data = None
    risk_color = "light"
    risk_level = "Chưa có dự đoán" 
    
    if request.method == 'POST':
        # Lấy email từ form. Dùng RECEIVER_EMAIL nếu người dùng không nhập.
        user_email_input = request.form.get("user_email")
        receiver_for_alert = user_email_input if user_email_input and '@' in user_email_input else RECEIVER_EMAIL

        if 'file' not in request.files or request.files['file'].filename == "":
            label_final = "Vui lòng chọn file ảnh."
        else:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                if file_path.lower().endswith((".tif", ".tiff")):
                    img_array = tiff.imread(file_path).astype(np.uint8)
                    img_array = np.nan_to_num(img_array, nan=0, posinf=0, neginf=0).astype(np.uint8) 

                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.ndim == 3 and img_array.shape[2] > 3:
                        img_array = img_array[..., :3] 

                    pil_img = Image.fromarray(img_array)
                    
                else:
                    pil_img = Image.open(file_path).convert("RGB")

                # Convert Pillow image sang Base64 để hiển thị
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_data = base64.b64encode(buffered.getvalue()).decode()
                
                # CHẠY DỰ ĐOÁN
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class_idx = torch.argmax(probs, dim=1).item()
                
                pred_label = class_names[pred_class_idx]
                pred_prob = probs.squeeze().cpu().numpy()[pred_class_idx]
                
                # === ÁP DỤNG LOGIC RỦI RO CUỐI CÙNG ===
                if pred_class_idx == 0:
                    risk_level = "Rất Thấp (An toàn)"
                    risk_color = "success"
                elif pred_class_idx == 1:
                    risk_level = "Trung Bình (Cần chú ý)"
                    risk_color = "warning"
                else: 
                    risk_level = "CAO (Cần cảnh báo)"
                    risk_color = "danger"
                    
                if pred_class_idx == 2:  # Chỉ gửi cảnh báo cho 'heavy_rain'
                    
                    subject = f"[CẢNH BÁO KHẨN CẤP] Phát hiện Nguy cơ {risk_level}!"
                    if send_alert_email(subject, pred_label, risk_level, pred_prob, img_data, receiver_for_alert):
                        print(f"Đã gửi email cảnh báo tới {receiver_for_alert}.")
                    else:
                        print(f"Gửi email cảnh báo thất bại tới {receiver_for_alert}.")
                
                label_final = f"Dự đoán: {pred_label} | Nguy cơ: {risk_level} (Độ tự tin: {pred_prob*100:.2f}%)"

            except Exception as e:
                label_final = f"Lỗi xử lý ảnh: {e}"
                print("Lỗi xử lý ảnh:", e)
                # Đảm bảo các biến rủi ro được set cho thông báo lỗi
                risk_color = "secondary" 
                risk_level = "Lỗi kỹ thuật"


    return render_template("index.html", 
                           label=label_final, 
                           risk_color=risk_color, 
                           risk_level=risk_level, 
                           img_data=img_data) 
#Trang xem dữ liệu mưa
@app.route('/rainfall', methods=['GET', 'POST'])
def getRain():
    years = sorted(df['year'].unique())
    selected_year = None
    year_data = None

    if request.method == "POST":
        selected_year = int(request.form.get("year"))
        year_data = df[df['year'] == selected_year].sort_values('date')

    return render_template("rainfall.html", years=years, selected_year=selected_year, year_data=year_data)

app.run()

    