# Flask Code to put the UI together to upload image and obtain captions from the images by using the blip and vit-gpt2 image to text models.


from flask import Flask, request, render_template
import os
import caption_generation_blip
import caption_generation_vit_gpt2

app = Flask(__name__ )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get the uploaded file from the request object
    file = request.files['image']
    
    # Save the file to the specified directory
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    
    
    # Process the uploaded file and generate a caption
    
     #break down the captions from the list caption
    breakdown_captions = "BLIP:Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation model results: \n\n "
    
    # This model is The BLIP Sales force model
    caption_blip_salesforce = caption_generation_blip.generate_caption_BLIP_SALES_FORCE(file)
    
    
    caption_vit_gpt2 = caption_generation_vit_gpt2.predict_step([file])
   

    
    for i in caption_blip_salesforce:
        breakdown_captions =  breakdown_captions + i + "\n\n"
    breakdown_captions =  breakdown_captions +"Vit-gpt2-image-captioning model results: "+ "\n\n"
    for i in caption_vit_gpt2:
        breakdown_captions =  breakdown_captions + i + "\n\n"
    
    #Render the output in the HTML template
    return render_template('output.html', IMAGE = '/static/'+file.filename , caption=breakdown_captions)


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = '/Users/software/Desktop/Listed/static'
    app.run()