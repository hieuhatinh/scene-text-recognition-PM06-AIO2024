import streamlit as st
import requests
from PIL import Image
import io


base_url = 'http://localhost:8000'

st.set_page_config(layout='wide', page_title='Scene Text Recognition Project')


def process_image_upload(file_image, base_url):
    try:
        files = {'file': (file_image.name, file_image, file_image.type)}
        response = requests.post(f'{base_url}/ocr/upload', files=files)
        predictions = response.headers.get('X-predictions', '[]')
        image = Image.open(io.BytesIO(response.content))
        return predictions, image
    except Exception as e:
        print(e)
        st.error(e)


def format_predictions(predictions):
    """Format predictions string into readable JSON"""
    try:
        # Safely evaluate the string representation of predictions
        predictions = eval(predictions)

        if not predictions:
            return "[]"

        # Format the JSON-like structure with indentation
        formatted_json = "[\n"
        for bbox, text, class_name, confidence in predictions:
            formatted_json += "  {\n"
            formatted_json += f"    \"bbox\": {bbox},\n"
            formatted_json += f"    \"class\": \"{class_name}\",\n"
            formatted_json += f"    \"confidence\": {confidence:.2f},\n"
            formatted_json += f"    \"text\": \"{text}\"\n"
            formatted_json += "  },\n"
        formatted_json = formatted_json.rstrip(",\n") + "\n]"

        return formatted_json
    except Exception as e:
        return f"Error formatting predictions: {str(e)}"


def main():
    st.header('Project Scene Text Recognition using CRNN and YOLOv11')

    tab1, tab2 = st.tabs(['Upload Image', 'Image From URL'])

    with tab1:
        img_upload = st.file_uploader(
            'Upload an image', type=['png', 'jpeg', 'jpg'])

        if img_upload:
            process_button = st.button('Process Image')
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Origin Image')
                st.image(img_upload, use_container_width=True)
                st.text(img_upload.name)

            with col2:
                st.subheader('Processed Image')

                if process_button:
                    with st.spinner('Đang xử lý...'):
                        predictions, image = process_image_upload(
                            img_upload, base_url=base_url)
                        st.image(image, use_container_width=True)
                        st.code(format_predictions(predictions))


if __name__ == '__main__':
    main()
