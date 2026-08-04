import cv2
import numpy as np

def convert_binary_to_red_alpha_opencv(input_path, output_path, alpha_value=75):
    """
    Chuyển đổi hình nhị phân sang màu Đỏ với Alpha tùy chỉnh bằng OpenCV.
    Vùng trắng -> Đỏ (Alpha=75), Vùng đen -> Trong suốt (Alpha=0).
    """
    # 1. Đọc ảnh dưới dạng ảnh xám (grayscale)
    binary_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if binary_img is None:
        print(f"Lỗi: Không thể tải ảnh từ {input_path}")
        return

    # 2. Đảm bảo ảnh là nhị phân hoàn toàn (0 hoặc 255)
    # Các pixel > 127 sẽ thành 255 (trắng), còn lại thành 0 (đen).
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

    # 3. Lấy kích thước ảnh
    height, width = binary_img.shape

    # 4. Tạo các kênh màu RGB
    # Kênh R (Đỏ): Giữ nguyên vùng trắng của ảnh nhị phân (255 là Đỏ hoàn toàn trong RGB)
    red_channel = binary_img
    
    # Kênh G (Xanh lá) và B (Xanh dương): Chuyển hoàn toàn thành 0
    green_channel = np.zeros((height, width), dtype=np.uint8)
    blue_channel = np.zeros((height, width), dtype=np.uint8)

    # 5. Tạo kênh Alpha (Độ trong suốt)
    # Khởi tạo kênh Alpha toàn 0 (trong suốt hoàn toàn)
    alpha_channel = np.zeros((height, width), dtype=np.uint8)
    
    # Tại những vị trí ảnh nhị phân là trắng (255), set Alpha = alpha_value
    alpha_channel[binary_img == 255] = alpha_value

    # 6. Gộp các kênh lại thành ảnh BGRA (OpenCV dùng thứ tự BGR)
    rgba_img = cv2.merge((blue_channel, green_channel, red_channel, alpha_channel))

    # 7. Lưu ảnh (phải lưu định dạng hỗ trợ Alpha như .png)
    cv2.imwrite(output_path, rgba_img)
    print(f"Đã chuyển đổi và lưu tại: {output_path}")

# --- Sử dụng ---
# Tạo một ảnh nhị phân giả lập để test nếu bạn chưa có file
# test_binary = np.zeros((200, 200), dtype=np.uint8)
# cv2.rectangle(test_binary, (50, 50), (150, 150), 255, -1) # Hình vuông trắng ở giữa
# cv2.imwrite('input_binary.png', test_binary)

input_file = r'G:\NewBrukerAI\Dataset\UNetDataTSD\result1\Untitled7.png'   # Đường dẫn file ảnh nhị phân của bạn
output_file =  r'G:\NewBrukerAI\Dataset\UNetDataTSD\result1\Untitled7_argb.png'

convert_binary_to_red_alpha_opencv(input_file, output_file, alpha_value=75)