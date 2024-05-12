import cv2

def main():
    # Khởi tạo camera
    cap = cv2.VideoCapture(2)

    # Kiểm tra xem camera đã mở thành công hay chưa
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        # Kiểm tra nếu không thể đọc frame
        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        # Hiển thị frame trên cửa sổ
        cv2.imshow('Camera', frame)

        # Đợi 1 phím bất kỳ được nhấn trong 1ms
        # Nếu phím 'q' được nhấn, thoát khỏi vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
