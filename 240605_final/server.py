# import .py
import ocrrun
import summarize

# import package
import socket
import os
import shutil
import time

def resetall(save_dir):
    print("=====Delete files signal received.=====")
    # 디렉토리 내 모든 파일 삭제
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"!!!!!Failed to delete {file_path}. Reason: {e}!!!!!")
    print("=====All files deleted successfully.=====")

def save_img(save_dir, conn):
    print("=====File transfer signal received.=====")
    # file_name_length = int.from_bytes(conn.recv(4), 'big')
    # file_name_bytes = conn.recv(file_name_length)
    # file_name = file_name_bytes.decode('utf-8')
    # file_extension = os.path.splitext(file_name)[1]

    unique_file_name = f"scan.jpg"
    file_path = os.path.join(save_dir, unique_file_name)

    with open(file_path, 'wb') as f:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            f.write(data)
    print(f"=====File received successfully and saved as {file_path}=====")

def handle_signal(signal, save_dir, conn):
    try:
        if signal == '1':
            resetall(save_dir)
            print(f'===== signal 1 =====')
            conn.sendall(b'All files deleted successfully.')
        elif signal in {'3', '4', '5'}:
            save_img(save_dir, conn)
            print(f'===== signal {signal} =====')
            result_ocr = ocrrun.main(int(signal) - 2)
            if result_ocr:
                conn.send(b'complete')
            else:
                print('!!!!!!signal receive fail!!!!!')
        elif signal == '7':
            print(f'===== signal 7 =====')
            # text_path = os.path.join(save_dir, 'text.txt')
            # if not os.path.exists(text_path):
            #     raise FileNotFoundError(f"{text_path} not found.")
            result_smz = summarize.summarizing()
            print(result_smz)
            conn.sendall(result_smz.encode('utf-16'))
        elif signal == '9':
            time.sleep(1)
            print("===== test signal =====")
            conn.sendall(b'test response')
        else:
            print("?????Unknown signal received.?????")
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        print(f"Error: {error_msg}")
        conn.sendall(error_msg.encode('utf-16'))

def start_server(host='0.0.0.0', port=8080, save_dir='received_files'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    resetall(save_dir)
    print('All files deleted successfully')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"**********Server listening on {host}:{port}**********")

    while True:
        conn, addr = server_socket.accept()
        print(f"=====Connection=====")

        with conn:
            signal = conn.recv(1).decode('utf-8')
            handle_signal(signal, save_dir, conn)

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("**********Server is shutting down.**********")
    finally:
        server_socket.close()
        print("**********Socket closed.**********")