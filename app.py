from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = 'snoopy'


app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = '/tmp/flask_sessions'

os.makedirs('/tmp/flask_sessions', exist_ok=True)
ada_pipeline_m1  = joblib.load('models/exploration_pipeline_m1.pkl')  # ~50% real labels
ada_pipeline_m2  = joblib.load('models/exploration_pipeline_m2.pkl')  # ~80% profile-based
knn              = joblib.load('models/guessing_knn.pkl')
preprocessor_g   = joblib.load('models/guessing_preprocessor.pkl')
career_df        = joblib.load('models/career_df.pkl')

CAREER_INFO = {
    'Technology & Engineering': {
        'emoji': '👓🐶💻',
        'snoopy': None,
        'message': 'Technology & Engineering is your best-fit career field! You have a natural affinity for systems, logic, and building things that work. Whether it\'s writing code, designing systems, or solving complex technical puzzles — this world was made for someone like you.',
        'why': 'Your high conscientiousness, openness to new ideas, and comfort with abstract thinking point strongly toward tech. People who thrive here love structure with creativity — they want to build things that actually work and can be improved.',
        'jobs': ['Software Engineer', 'Data Engineer', 'Network Engineer', 'Cybersecurity Analyst', 'DevOps Engineer', 'Systems Architect'],
    },
    'Data & Analytics': {
        'emoji': '📊🐶🔍',
        'snoopy': None,
        'message': 'Data & Analytics is your best-fit career field! You have a natural talent for finding patterns in numbers and turning raw information into meaningful insights. The world needs people who can make sense of the data flood — and that person is you.',
        'why': 'Your high conscientiousness, moderate extraversion, and love of logical problem-solving are hallmarks of great data professionals. You enjoy digging into the "why" behind things rather than just accepting surface-level answers.',
        'jobs': ['Data Scientist', 'Business Analyst', 'ML Engineer', 'Data Analyst', 'Quantitative Researcher', 'BI Developer'],
    },
    'Creative & Design': {
        'emoji': '🎨🐶🖌️',
        'snoopy': None,
        'message': 'Creative & Design is your best-fit career field! You see the world differently — as something to be shaped, imagined, and made more beautiful. Your ability to communicate ideas visually or through storytelling is a rare and powerful gift.',
        'why': 'Your high openness score and creative interests reveal someone who thinks in pictures, stories, and experiences. Creative professionals thrive on turning abstract ideas into something tangible that moves people emotionally.',
        'jobs': ['Graphic Designer', 'UX Designer', 'Content Creator', 'Animator', 'Art Director', 'Copywriter', 'Filmmaker'],
    },
    'Business & Management': {
        'emoji': '💼🐶📈',
        'snoopy': None,
        'message': 'Business & Management is your best-fit career field! You have a natural instinct for leadership, strategy, and getting things done through people. You see opportunities where others see obstacles, and you know how to bring a team together to achieve big goals.',
        'why': 'Your high extraversion and conscientiousness, combined with strong business interest, are classic traits of effective leaders and managers. You enjoy influence, decision-making, and the challenge of building something bigger than yourself.',
        'jobs': ['Product Manager', 'Marketing Manager', 'Entrepreneur', 'Business Analyst', 'Financial Advisor', 'HR Manager'],
    },
    'Education & Social Impact': {
        'emoji': '📚🐶✏️',
        'snoopy': None,
        'message': 'Education & Social Impact is your best-fit career field! You genuinely care about people and believe in making the world a better place through knowledge, empathy, and service. Your calling is to lift others up and leave every situation better than you found it.',
        'why': 'Your high agreeableness and people interest reveal someone driven by purpose rather than just profit. You feel most fulfilled when your work directly helps or improves someone\'s life — teaching, counseling, or community building.',
        'jobs': ['Teacher', 'School Counselor', 'Social Worker', 'NGO Manager', 'Trainer', 'Professor', 'Community Organizer'],
    },
    'Healthcare & Science': {
        'emoji': '🏥🐶💊',
        'snoopy': None,
        'message': 'Healthcare & Science is your best-fit career field! You combine precision with compassion — you want to understand how things work at the deepest level and use that knowledge to help people live better lives. The world depends on people with your dedication.',
        'why': 'Your high conscientiousness, strong people interest, and disciplined study habits align perfectly with healthcare and scientific careers. These fields demand both rigorous analytical thinking and deep human empathy — exactly your combination.',
        'jobs': ['Doctor', 'Nurse', 'Pharmacist', 'Medical Researcher', 'Lab Scientist', 'Dentist', 'Nutritionist'],
    },
    'Other': {
        'emoji': '🌟🐶🗺️',
        'snoopy': None,
        'message': 'Your career path is unique and doesn\'t fit neatly into one box — and that\'s actually wonderful! You may be destined for an emerging field, a cross-disciplinary role, or something that doesn\'t even have a name yet.',
        'why': 'Your answers show a diverse mix of interests and traits. This often means you\'re a generalist who can bridge multiple fields — a rare and increasingly valuable type of professional in a complex world.',
        'jobs': ['Consultant', 'Entrepreneur', 'Researcher', 'Policy Analyst', 'Innovation Manager', 'Interdisciplinary Specialist'],
    },
}

GUESS_QUESTIONS = [
    {"key": "paid", "text": "💰 Do you get paid for this work?", "options": ["yes", "no", "sometimes"]},
    {"key": "physical_presence", "text": "🏢 Do you need to be physically present at a specific place?", "options": ["yes", "no", "sometimes"]},
    {"key": "location_specific", "text": "📍 Is the job tied to one specific location or city?", "options": ["yes", "no"]},
    {"key": "manual_work", "text": "🔧 Does the job involve physical or manual work?", "options": ["yes", "no"]},
    {"key": "creative_output", "text": "🎨 Is the work primarily creative in nature?", "options": ["yes", "no"]},
    {"key": "technical_work", "text": "💻 Is the work technical (code, machines, engineering)?", "options": ["yes", "no"]},
    {"key": "people_interaction", "text": "🤝 How much do you work directly with people?", "options": ["low", "medium", "high"]},
    {"key": "scale", "text": "🌍 How large is the scale or reach of your work?", "options": ["small", "medium", "large", "massive"]},
    {"key": "uniqueness_level", "text": "🦄 How niche or unique is the job?", "options": ["low", "medium", "high"]},
    {"key": "automation_risk", "text": "🤖 How easily could AI replace this job?", "options": ["low", "medium", "high"]},
    {"key": "social_impact", "text": "❤️ Does the job directly help or improve people's lives?", "options": ["low", "medium", "high"]},
    {"key": "product_type", "text": "📦 What do people receive from your work?", "options": ["physical", "digital", "service", "content", "experience"]},
    {"key": "customer_type", "text": "👥 Who primarily benefits from your work?", "options": ["individuals", "companies", "organizations", "brands", "platform"]},
    {"key": "domain", "text": "🗂️ Which domain best describes your work?", "options": ["technology", "music", "automotive", "religion", "social media", "design", "entertainment", "marketing", "education", "non-profit"]},
    {"key": "typical_age_min", "text": "🎓 At roughly what age do people typically start this career?", "options": ["16", "18", "21", "25", "30"]},
]

def get_guess_message(career_name):
    return (
        f"Based on your answers, the AI identified you as a {career_name}! "
        f"Your combination of work style, domain, and how you interact with the world "
        f"closely matches real profiles of people in this career."
    )

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guess')
def guess_start():
    session.clear()
    session['guess_answers'] = {}
    session['guess_step'] = 0
    q = GUESS_QUESTIONS[0]
    return render_template('guess.html', question=q, step=1, total=len(GUESS_QUESTIONS))

@app.route('/guess/answer', methods=['POST'])
def guess_answer():
    answers = session.get('guess_answers', {})
    step = session.get('guess_step', 0)
    q = GUESS_QUESTIONS[step]

    answers[q['key']] = request.form.get('answer')
    step += 1
    session['guess_answers'] = answers
    session['guess_step'] = step

    if step >= len(GUESS_QUESTIONS):
        return _guess_predict()

    return render_template('guess.html',
                           question=GUESS_QUESTIONS[step],
                           step=step + 1,
                           total=len(GUESS_QUESTIONS))

def _guess_predict():
    answers = dict(session.get('guess_answers', {}))
    answers['typical_age_min'] = int(answers.get('typical_age_min', 21))

    col_order = ['typical_age_min', 'paid', 'physical_presence', 'location_specific',
                 'manual_work', 'creative_output', 'technical_work', 'people_interaction',
                 'scale', 'uniqueness_level', 'automation_risk', 'social_impact',
                 'product_type', 'customer_type', 'domain']

    user_df = pd.DataFrame([answers])[col_order]
    user_enc = preprocessor_g.transform(user_df)

    predicted = knn.predict(user_enc)[0]
    distances, indices = knn.kneighbors(user_enc)
    related = career_df.iloc[indices[0]]['career_name'].tolist()

    career_msg = get_guess_message(predicted)

    return render_template('guess_result.html',
                           career=predicted,
                           career_msg=career_msg,
                           related=related,
                           snoopy_img=None)

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/explore/result', methods=['POST'])
def explore_result():
    f = request.form
    model = f.get('model', '1')

    def to_binary(val):
        return 1 if int(val) >= 3 else 0

    user_data = {
        'tech_interest': to_binary(f.get('tech_interest', 0)),
        'data_interest': to_binary(f.get('data_interest', 0)),
        'creative_interest': to_binary(f.get('creative_interest', 0)),
        'business_interest': to_binary(f.get('business_interest', 0)),
        'people_interest': to_binary(f.get('people_interest', 0)),
        'Extraversion': float(f.get('extraversion', 0.5)),
        'Neuroticism': float(f.get('neuroticism', 0.5)),
        'Agreeableness': float(f.get('agreeableness', 0.5)),
        'Conscientiousness': float(f.get('conscientiousness', 0.5)),
        'Openness': float(f.get('openness', 0.5)),
        'study_hours_per_day': min(float(f.get('study_hours', 4)) / 12, 1.0),
        'attendance_percentage': min(float(f.get('attendance', 80)) / 100, 1.0),
        'motivation_level': min(float(f.get('motivation', 3)) / 5, 1.0),
        'time_management_score': min(float(f.get('time_mgmt', 3)) / 5, 1.0),
        'extracurricular_participation': int(f.get('extracurricular', 0)),
        'stress_level': min(float(f.get('stress', 3)) / 5, 1.0),
        'learning_style': int(f.get('learning_style', 0)),
    }

    user_df = pd.DataFrame([user_data])
    pipeline = ada_pipeline_m1 if model == '1' else ada_pipeline_m2
    probs = pipeline.predict_proba(user_df)[0]
    classes = pipeline.classes_

    top_career = classes[np.argmax(probs)]
    info = CAREER_INFO.get(top_career, CAREER_INFO['Other'])

    return render_template('explore_result.html',
                           model=model,
                           top_career=top_career,
                           career_emoji=info['emoji'],
                           snoopy_img=None,
                           career_message=info['message'],
                           why_message=info['why'],
                           sample_jobs=info['jobs'])

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
