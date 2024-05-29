# import .py
import ocrrun
import summarize

# import package
import socket
import os
import shutil
import time

def ressetall(save_dir):
    print("Delete files signal received.")
    # 디렉토리 내 모든 파일 삭제
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print("All files deleted successfully.")


def save_img(save_dir, conn):
    print("File transfer signal received.")
    # 파일명 길이 수신 (4 바이트)
    file_name_length = int.from_bytes(conn.recv(4), 'big')
    # 파일명 수신
    file_name_bytes = conn.recv(file_name_length)
    file_name = file_name_bytes.decode('utf-8')
    file_extension = os.path.splitext(file_name)[1]

    # 고유한 파일명 생성 (여기서는 항상 scan.확장자)
    unique_file_name = f"scan{file_extension}"
    file_path = os.path.join(save_dir, unique_file_name)

    with open(file_path, 'wb') as f:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            f.write(data)
    print(f"File received successfully and saved as {file_path}")


# run server program
def start_server(host='0.0.0.0', port=8080, save_dir='received_files'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")


    while True:
        conn, addr = server_socket.accept()
        print(f"Connection")

        with conn:
            # 신호 수신
            signal = conn.recv(1).decode('utf-8')
            if signal == '1':
                ressetall(save_dir)
                file_count = 0
                conn.sendall(b'All files deleted successfully.')
            elif signal == '3':
                save_img(save_dir, conn)
                ocrrun.main()
            elif signal == '4':
                save_img(save_dir, conn)
            elif signal == '5':
                save_img(save_dir, conn)
            elif signal == '7':
                result_smz = summarize.summarizing()
                conn.sendall(result_smz.encode('utf-8'))
            elif signal == '9':
                time.sleep(1)
                print("input test signal 3")
                conn.sendall(b'test response')
            else:
                print("Unknown signal received.")



if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("Server is shutting down.")
    finally:
        server_socket.close()
        print("Socket closed.")

