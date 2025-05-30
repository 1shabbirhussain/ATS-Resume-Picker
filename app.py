import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import openai
from rembg import remove
import io
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def remove_background(image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    
    # Remove background
    output_data = remove(img_data)
    
    # Save the result
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"nobg_{os.path.basename(image_path)}")
    with open(output_path, 'wb') as f:
        f.write(output_data)
    
    return output_path

def generate_image_with_dalle(template_path, person_image_path, name, position_image="center-right", position_text="center-left"):
    # Open the template image
    template = Image.open(template_path).convert("RGBA")
    template_width, template_height = template.size
    
    # Open the person's image (with background removed)
    person_img = Image.open(person_image_path).convert("RGBA")
    
    # Resize person's image to be bigger (e.g., 60% of template width)
    new_person_width = int(template_width * 1)
    person_img.thumbnail((new_person_width, template_height))  # keep aspect ratio
    
    # Calculate position for person image (center-right)
    x_person = template_width - person_img.width + 120  # 50 px padding from right
    y_person = (template_height - person_img.height) // 2  # vertically centered
    
    # Create a copy of the template to modify
    final_image = template.copy()
    
    # Paste the person's image onto the template (with alpha mask for transparency)
    final_image.paste(person_img, (x_person, y_person), person_img)
    
    # Prepare to draw text
    draw = ImageDraw.Draw(final_image)
    
    # Calculate font size bigger than before
    font_size = int(template_width / 8)  # bigger than before
    
    # Load font with fallback
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("impact.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("timesbd.ttf", font_size)
            except:
                font = ImageFont.load_default(font_size)
    
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate position for text (center-left)
    x_text = 100  # 50 px padding from left
    y_text = (template_height - text_height) // 2  # vertically centered
    
    # Draw blackish semi-transparent background rectangle behind text
    rectangle_padding = 20
    rectangle_x0 = x_text - rectangle_padding
    rectangle_y0 = y_text - rectangle_padding
    rectangle_x1 = x_text + text_width + rectangle_padding
    rectangle_y1 = y_text + text_height + rectangle_padding
    
    # Create semi-transparent black rectangle
    # rectangle_color = (0, 0, 0, 180)  # black with alpha 180/255
    
    # To draw semi-transparent shapes, we use an overlay image
    overlay = Image.new('RGBA', final_image.size, (0,0,0,0))
    overlay_draw = ImageDraw.Draw(overlay)
    # overlay_draw.rectangle([rectangle_x0, rectangle_y0, rectangle_x1, rectangle_y1],)
    
    # Composite overlay with final_image
    final_image = Image.alpha_composite(final_image, overlay)
    
    # Draw the text on top in gold color
    draw = ImageDraw.Draw(final_image)
    draw.text((x_text, y_text), name, font=font, fill=(255, 255, 255))
    
    # Save the final image as PNG with transparency preserved
    output_filename = f"generated_{uuid.uuid4().hex}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    final_image.convert("RGB").save(output_path, quality=95)
    
    return output_path



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'template' not in request.files or 'person_image' not in request.files:
        return jsonify({"error": "Missing files"}), 400
    
    template_file = request.files['template']
    person_file = request.files['person_image']
    name = request.form.get('name', '')
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    if template_file.filename == '' or person_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if template_file and allowed_file(template_file.filename) and person_file and allowed_file(person_file.filename):
        # Save uploaded files
        template_filename = secure_filename(template_file.filename)
        person_filename = secure_filename(person_file.filename)
        
        template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
        person_path = os.path.join(app.config['UPLOAD_FOLDER'], person_filename)
        
        template_file.save(template_path)
        person_file.save(person_path)
        
        # Remove background from person's image
        nobg_person_path = remove_background(person_path)
        
        # Generate final image
        try:
            final_image_path = generate_image_with_dalle(template_path, nobg_person_path, name)
            final_filename = os.path.basename(final_image_path)
            return jsonify({"image_url": f"/uploads/{final_filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)