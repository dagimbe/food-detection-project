Create virtual environment:
python -m venv venv
.\venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

to train the model 
python src/train.py

to evaluate the model 
python src/evaluate.py

Run the Streamlit App
streamlit run app.py
