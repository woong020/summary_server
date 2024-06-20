# import .py
import ocrrun
import summarize

# import package
import socket
import os
import shutil
import time

# delete signal 수신 시 아래 메소드 실행
def resetall(save_dir):
    print("=====Delete files signal received.=====")

    # save_dir 인 코드 실행 경로 내 receive_files 디렉토리 내용 전체 삭제
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

# send file signal 수신 시 아래 메소드 실행
def save_img(save_dir, conn):
    print("=====File transfer signal received.=====")

    # 코드 실행 경로 내 receive_files 디렉토리에 scan.jpg 저장
    file_name = f"scan.jpg"
    file_path = os.path.join(save_dir, file_name)

    # data 수신이 완료될 때 까지 수신받음
    with open(file_path, 'wb') as f:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            f.write(data)
    print(f"=====File received successfully and saved as {file_path}=====")

# signal 수신 시 실행할 메소드 제어
# signal 1 : resetall (파일 전체 삭제)
# signal 3, 4, 5 : save_img (파일 저장), size 별로 수신되는 signal 다름
# signal 7 : 저장된 text.txt 파일을 요약
# signal 9 : Easter eggs
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
            result_smz = summarize.summarizing()
            print(result_smz)
            conn.sendall(result_smz.encode('utf-16'))
        elif signal == '9':
            time.sleep(1)
            print("===== test signal =====")
            conn.sendall(b'the one team =|hw|jm|sj|hd|gw|hs|jy|=')
        else:
            print("?????Unknown signal received.?????")
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        print(f"Error: {error_msg}")
        conn.sendall(error_msg.encode('utf-16'))

# socket 생성 및 listen 실행 메소드. 전체 IP에 대해 8080 포트 오픈, save_dir 지정
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