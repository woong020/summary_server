# 전역 변수를 사용하여 save_count를 관리합니다.
save_count = 0

def ocr_apply(pil_image, output_dir):
    global save_count
    
    # 이미지에서 텍스트를 추출합니다.
    result = ocr.ocr(pil_image, cls=True)

    # 추출된 텍스트를 한 문장으로 이어서 출력합니다.
    text = ''
    for line in result:
        text += ' '.join([word[1][0] for word in line]) + ' '

    # 파일 이름이 중복되지 않도록 확인하고 저장
    base_filename = "text"
    extension = ".txt"
    output_filename = os.path.join(output_dir, f"{base_filename}{extension}")

    # 파일이 존재하지 않으면 새로운 파일 생성
    if not os.path.exists(output_filename):
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f'Extracted text saved to: {output_filename}')
        save_count = 1  # 파일 생성 후 첫 번째 저장으로 설정
    else:
        if save_count == 0:
            # 초기 save_count 설정
            with open(output_filename, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    save_count = content.count("Extracted text saved to") + content.count("Extracted text appended to") + content.count("Extracted text overwritten to")

        # 2회 저장 후 save_count를 리셋
        if save_count % 2 == 0:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f'Extracted text overwritten to: {output_filename}')
        else:
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(text)
            print(f'Extracted text appended to: {output_filename}')
        
        save_count = (save_count + 1) % 2  # save_count를 2회마다 리셋