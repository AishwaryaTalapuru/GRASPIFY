import fitz
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import whisper
import json
import traceback
import os
from sentence_transformers import SentenceTransformer
import torch


class Input:
    def __init__(self, audio_path="", images_path="", text_path="", data_path="transcription/data.json"):
        self.data_path = data_path
        self.audio_path = audio_path
        self.images_path = images_path
        self.text_path = text_path
        # Clear the transcription/data.json file and initialize it with an empty list
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
            print("JSON file deleted.")

    def save_to_json(self, data):
        print(f"##### Saving Multimodal Data to {self.data_path} #####")
        # Load existing data as a list
        try:
            with open(self.data_path, 'r') as file:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Append new data
        existing_data.extend(data)

        # Write back to file
        with open(self.data_path, 'w') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)



    def extract_preprocess_pdf_chunks(self):
        """
        Extracts text from a PDF, preprocesses it line-by-line, and generates embeddings for each non-empty line.
        Returns a list of dictionaries with metadata and embeddings.
        """
        import logging
        from tqdm import tqdm
        import traceback

        logging.basicConfig(level=logging.INFO)
        print("##### Preprocessing TEXT Data #####")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            doc = fitz.open(self.text_path)
            chunks = []
            for i, page in enumerate(tqdm(doc, desc="Processing pages")):
                text = page.get_text("text")
                lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
                if not lines:
                    continue
                embeddings = model.encode(lines, convert_to_tensor=False)
                for line_no, (line, embedding) in enumerate(zip(lines, embeddings), start=1):
                    logging.info(f"page : {i+1} line_no : {line_no}, line : {line}")
                    chunks.append({
                        "type": "text",
                        "subtype": "pdf",
                        "page": i + 1,
                        "line_no": line_no,
                        "content": line,
                        "embedding": embedding.tolist()
                    })
            self.save_to_json(chunks)
            return chunks
        except Exception as e:
            logging.error("Error occurred while extracting PDF chunks:")
            traceback.print_exc()
            chunks = [{
                "type": "text",
                "subtype": "pdf",
                "exception": str(e)
            }]
            self.save_to_json(chunks)
            return chunks


    def caption_preprocess_image(self):
        print(f"##### Preprocessing IMAGE CAPTION Data #####")
        try:

            model_1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor_1 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            image_1 = Image.open(self.images_path)
            inputs_1 = processor_1(images=image_1, return_tensors="pt")
            outputs_1 = model_1.get_image_features(**inputs_1)
            embedding = outputs_1[0].detach().numpy()

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            image = Image.open(self.images_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

            img = Image.open(self.images_path)
            text = pytesseract.image_to_string(img)


            chunks = [{
                "type": "caption_of_image",
                "subtype": "caption",
                "filename": self.images_path,
                "content": caption, 
                "text_on_image": text, 
                "embedding": embedding.tolist()
            }]
            self.save_to_json(chunks)
        except Exception as e:
            print("Error occurred while extracting image caption chunks:")
            traceback.print_exc()
            chunks = [{
                "type": "caption_of_image",
                "subtype": "caption",
                "exception": str(e)
            }]
            self.save_to_json(chunks)        
        return chunks

    def transcribe_preprocess_audio(self):
        print(f"##### Preprocessing AUDIO Data #####")
        try:
            model = whisper.load_model("base")
            model_text = SentenceTransformer('all-MiniLM-L6-v2')
            result = model.transcribe(self.audio_path, verbose=False)
            chunks = []
            for segment in result['segments']:
                text = segment['text'].strip()
    
                if not text:
                    continue

                try:  # show first 30 chars
                    embedding = model_text.encode(text, convert_to_tensor=False)

                    chunks.append({
                        "type": "audio",
                        "subtype": "mp3",
                        "start": segment['start'],
                        "end": segment['end'],
                        "content": text,
                        "embedding": embedding.tolist()
                    })

                except Exception as e:
                    print(f"Error embedding segment '{text[:30]}...': {e}")
                    chunks.append({
                        "type": "audio",
                        "subtype": "mp3",
                        "start": segment['start'],
                        "end": segment['end'],
                        "content": text,
                        "embedding": f"Error: {str(e)}"
                    })
        except Exception as e:
            print("Error occurred while extracting audio chunks:")
            traceback.print_exc()
            chunks = [{
                "type": "audio",
                "subtype": "mp3",
                "exception": str(e)
            }]
        self.save_to_json(chunks)       

        return chunks


if __name__ == "__main__":
    path = "input_files/"
    audio_path = "audio.mp3"
    image_path = "image.png"
    text_path = "text.pdf"

    inp_obj = Input(
        audio_path=path + "audio/" + audio_path,
        images_path=path + "images/" + image_path,
        text_path=path + "text/" + text_path
    )

    #print("######## Extracting PDF ########")
    inp_obj.extract_preprocess_pdf_chunks()

    #print("######## Generating Caption ########")
    inp_obj.caption_preprocess_image()

    #print("######## Transcribing Audio ########")
    inp_obj.transcribe_preprocess_audio()


    """
    print("######## Final JSON Output ########")
    with open(inp_obj.data_path, 'r') as f:
        print(json.load(f))
    """
