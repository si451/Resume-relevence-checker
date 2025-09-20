import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time

# Import our scoring modules
from scoring.parser import DocumentParser
from scoring.skill_extractor import SkillExtractor
from scoring.scoring import ResumeScorer
from scoring.feedback import FeedbackGenerator

# Set page config
st.set_page_config(
    page_title="Resume Relevance Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
if 'jd_skills' not in st.session_state:
    st.session_state.jd_skills = {"must_have": [], "good_to_have": []}

def load_tasks():
    """Load tasks from .cursor_tasks.json"""
    try:
        with open('.cursor_tasks.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tasks": [], "last_updated": "Never"}

def main():
    st.title("📄 Resume Relevance Checker")
    st.markdown("Upload job descriptions and resumes to get AI-powered relevance scores and improvement suggestions.")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Job Description")
        
        # JD Upload or Text Input
        jd_option = st.radio("Choose JD input method:", ["Upload PDF", "Paste Text"])
        
        if jd_option == "Upload PDF":
            jd_file = st.file_uploader("Upload Job Description (PDF)", type=['pdf'])
            if jd_file:
                # Save uploaded file temporarily
                temp_jd_path = f"data/jds/{jd_file.name}"
                os.makedirs("data/jds", exist_ok=True)
                
                with open(temp_jd_path, "wb") as f:
                    f.write(jd_file.getbuffer())
                
                # Extract text from PDF
                parser = DocumentParser()
                extract_result = parser.extract_text(temp_jd_path)
                
                if extract_result["success"]:
                    st.session_state.jd_text = extract_result["text"]
                    st.success(f"Uploaded and processed: {jd_file.name}")
                    st.info(f"Extracted {len(extract_result['text'])} characters")
                else:
                    st.error(f"Failed to extract text from {jd_file.name}: {extract_result['error']}")
                    st.session_state.jd_text = ""
        else:
            st.session_state.jd_text = st.text_area(
                "Paste Job Description:",
                value=st.session_state.jd_text,
                height=200
            )
        
        # Skills extraction (placeholder)
        if st.session_state.jd_text and st.session_state.jd_text != "PDF content will be extracted here...":
            if st.button("Extract Skills"):
                # TODO: Implement skill extraction
                st.session_state.jd_skills = {
                    "must_have": ["Python", "SQL", "Machine Learning", "Pandas", "Data Modeling"],
                    "good_to_have": ["TensorFlow", "AWS", "Spark", "Tableau", "Git", "Docker", "NLP", "Communication"]
                }
                st.success("Skills extracted!")
        
        # Display extracted skills
        if st.session_state.jd_skills["must_have"]:
            st.subheader("Must-Have Skills")
            for skill in st.session_state.jd_skills["must_have"]:
                st.write(f"• {skill}")
            
            st.subheader("Good-to-Have Skills")
            for skill in st.session_state.jd_skills["good_to_have"]:
                st.write(f"• {skill}")
        
        # Show JD text preview if available
        if st.session_state.jd_text and st.session_state.jd_text.strip():
            with st.expander("📄 Job Description Preview"):
                preview_text = st.session_state.jd_text[:1000] + "..." if len(st.session_state.jd_text) > 1000 else st.session_state.jd_text
                st.text_area("Job Description Text", preview_text, height=200, disabled=True, label_visibility="collapsed")
        
        st.divider()
        
        # Resume Upload
        st.header("📁 Resume Upload")
        uploaded_resumes = st.file_uploader(
            "Upload Resumes (PDF/DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_resumes:
            st.success(f"Uploaded {len(uploaded_resumes)} resume(s)")
            for resume in uploaded_resumes:
                st.write(f"• {resume.name}")
        
        st.divider()
        
        # Scoring Configuration
        st.header("⚙️ Scoring Configuration")
        
        # Simple explanation
        with st.expander("ℹ️ How Scoring Works", expanded=False):
            st.markdown("""
            **🎯 Hard Match**: Exact keyword matching between resume and job description
            **🧠 Soft Match**: AI-powered semantic similarity analysis
            **📊 Final Score**: Weighted combination of both methods
            """)
        
        # Weight sliders
        col1, col2 = st.columns(2)
        
        with col1:
            hard_weight = st.slider(
                "Hard Match Weight", 
                0.0, 1.0, 0.6, 0.1,
                help="Weight for exact keyword matching"
            )
        
        with col2:
            soft_weight = st.slider(
                "Soft Match Weight", 
                0.0, 1.0, 0.4, 0.1,
                help="Weight for semantic similarity"
            )
        
        # Normalize weights
        total_weight = hard_weight + soft_weight
        if total_weight > 0:
            hard_weight = hard_weight / total_weight
            soft_weight = soft_weight / total_weight
        
        if st.button("🚀 Analyze Resumes", type="primary"):
            if not st.session_state.jd_text or not st.session_state.jd_text.strip():
                st.error("Please provide a job description first!")
            elif not uploaded_resumes:
                st.error("Please upload at least one resume!")
            else:
                # Initialize components
                parser = DocumentParser()
                skill_extractor = SkillExtractor(use_llm=True)
                scorer = ResumeScorer(hard_weight=hard_weight, soft_weight=soft_weight)
                feedback_generator = FeedbackGenerator(use_llm=True)
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract skills from JD
                    status_text.text("Extracting skills from job description...")
                    progress_bar.progress(10)
                    
                    jd_skills = skill_extractor.extract_skills(st.session_state.jd_text)
                    if not jd_skills["success"]:
                        st.error(f"Failed to extract skills: {jd_skills['error']}")
                        return
                    
                    must_have_skills = jd_skills["must_have"]
                    good_to_have_skills = jd_skills["good_to_have"]
                    
                    # Update session state
                    st.session_state.jd_skills = {
                        "must_have": must_have_skills,
                        "good_to_have": good_to_have_skills
                    }
                    
                    # Step 2: Process resumes
                    status_text.text("Processing resumes...")
                    progress_bar.progress(30)
                    
                    resumes_data = []
                    for i, resume_file in enumerate(uploaded_resumes):
                        # Save uploaded file temporarily
                        temp_path = f"data/resumes/{resume_file.name}"
                        os.makedirs("data/resumes", exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(resume_file.getbuffer())
                        
                        # Extract text
                        extract_result = parser.extract_text(temp_path)
                        if extract_result["success"]:
                            # Save extracted text
                            text_path = parser.save_extracted_text(
                                extract_result["text"], 
                                resume_file.name
                            )
                            
                            resumes_data.append({
                                "filename": resume_file.name,
                                "text": extract_result["text"],
                                "extracted_text_path": text_path
                            })
                        else:
                            st.warning(f"Failed to extract text from {resume_file.name}: {extract_result['error']}")
                    
                    if not resumes_data:
                        st.error("No resumes could be processed!")
                        return
                    
                    # Step 3: Score resumes
                    status_text.text("Scoring resumes...")
                    progress_bar.progress(60)
                    
                    results = scorer.score_multiple_resumes(
                        resumes_data, 
                        st.session_state.jd_text, 
                        must_have_skills, 
                        good_to_have_skills
                    )
                    
                    # Step 4: Generate feedback
                    status_text.text("Generating feedback...")
                    progress_bar.progress(80)
                    
                    feedback_results = feedback_generator.generate_batch_feedback(
                        results, 
                        st.session_state.jd_text
                    )
                    
                    # Combine results with feedback
                    final_results = []
                    for i, result in enumerate(results):
                        feedback = feedback_results[i] if i < len(feedback_results) else {}
                        final_results.append({
                            "resume_file": result["resume_file"],
                            "hard_pct": result["hard_pct"],
                            "soft_pct": result["soft_pct"],
                            "final_score": result["final_score"],
                            "verdict": result["verdict"],
                            "missing_skills": result["missing_skills"],
                            "grok_feedback": feedback.get("one_line_feedback", "No feedback available"),
                            "detailed_feedback": feedback.get("detailed_feedback", {}),
                            "extracted_text_path": result.get("extracted_text_path", "")
                        })
                    
                    st.session_state.results = final_results
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    st.success(f"Successfully analyzed {len(final_results)} resume(s)!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()
    
    # Main content area
    st.markdown("---")
    
    # Analysis Results Section
    st.header("📊 Analysis Results")
    
    # Show scoring configuration used
    if st.session_state.results:
        st.info(f"""
        **📋 Scoring Configuration Used:**
        - 🎯 **Hard Match Weight**: {hard_weight:.0%} (Keyword matching)
        - 🧠 **Soft Match Weight**: {soft_weight:.0%} (AI semantic similarity)
        - 📊 **Formula**: Final Score = ({hard_weight:.1f} × Hard %) + ({soft_weight:.1f} × Soft %)
        """)
    
    if st.session_state.results:
        # Show results in cards instead of table
        for i, result in enumerate(st.session_state.results):
            with st.container():
                st.markdown("---")
                
                # Create columns for the card
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**📄 {result['resume_file']}**")
                
                with col2:
                    # Color code the verdict
                    verdict_color = {
                        "High": "🟢",
                        "Medium": "🟡", 
                        "Low": "🔴"
                    }
                    st.write(f"{verdict_color.get(result['verdict'], '⚪')} **{result['verdict']}**")
                
                with col3:
                    st.write(f"**{result['final_score']:.1f}%**")
                
                with col4:
                    if st.button(f"View Details", key=f"details_{i}"):
                        st.session_state.selected_resume = i
                
                # Show progress bars
                col_progress1, col_progress2 = st.columns(2)
                with col_progress1:
                    st.progress(result['hard_pct']/100, text=f"Hard Match: {result['hard_pct']:.1f}%")
                with col_progress2:
                    st.progress(result['soft_pct']/100, text=f"Soft Match: {result['soft_pct']:.1f}%")
                
                # Show missing skills and feedback in expandable sections
                col_skills, col_feedback = st.columns(2)
                with col_skills:
                    if result['missing_skills']:
                        with st.expander("❌ Missing Skills", expanded=False):
                            for skill in result['missing_skills']:
                                st.write(f"• {skill}")
                    else:
                        st.success("✅ All required skills found!")
                
                with col_feedback:
                    with st.expander("🤖 AI Feedback", expanded=False):
                        st.write(result['grok_feedback'])
        
        # Detailed view for selected resume
        if len(st.session_state.results) > 0:
            st.markdown("---")
            st.subheader("📋 Detailed Analysis")
            
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Scores & Feedback", "📄 Resume Preview", "📋 Job Description"])
            
            selected_resume = st.selectbox(
                "Select a resume for detailed analysis:",
                options=range(len(st.session_state.results)),
                format_func=lambda x: st.session_state.results[x]["resume_file"]
            )
            
            if selected_resume is not None:
                result = st.session_state.results[selected_resume]
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Scoring Breakdown")
                        
                        # Show individual scores
                        col_hard, col_soft = st.columns(2)
                        with col_hard:
                            st.metric("🎯 Hard Match", f"{result['hard_pct']:.1f}%", 
                                    help="Exact keyword matches found in resume")
                        with col_soft:
                            st.metric("🧠 Soft Match", f"{result['soft_pct']:.1f}%", 
                                    help="AI semantic similarity score")
                        
                        # Show final score with calculation
                        st.metric("📊 Final Score", f"{result['final_score']:.1f}%", 
                                help=f"Calculated as: ({hard_weight:.1f} × {result['hard_pct']:.1f}%) + ({soft_weight:.1f} × {result['soft_pct']:.1f}%) = {result['final_score']:.1f}%")
                        
                        # Visual progress bars
                        st.subheader("📈 Score Visualization")
                        
                        # Hard match bar
                        hard_progress = result['hard_pct'] / 100
                        st.progress(hard_progress, text=f"Hard Match: {result['hard_pct']:.1f}%")
                        
                        # Soft match bar  
                        soft_progress = result['soft_pct'] / 100
                        st.progress(soft_progress, text=f"Soft Match: {result['soft_pct']:.1f}%")
                        
                        # Final score bar
                        final_progress = result['final_score'] / 100
                        st.progress(final_progress, text=f"Final Score: {result['final_score']:.1f}%")
                        
                        # Verdict with color coding
                        verdict_color = {
                            "High": "🟢",
                            "Medium": "🟡", 
                            "Low": "🔴"
                        }
                        st.write(f"**Verdict:** {verdict_color.get(result['verdict'], '⚪')} {result['verdict']}")
                        
                        if result['missing_skills']:
                            st.subheader("❌ Missing Skills")
                            for skill in result['missing_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.success("✅ **All required skills found!**")
                    
                    with col2:
                        st.subheader("🤖 AI Feedback")
                        st.info(result['grok_feedback'])
                        
                        # Add detailed feedback if available
                        if 'detailed_feedback' in result:
                            detailed = result['detailed_feedback']
                            if detailed.get('success'):
                                st.subheader("📝 Detailed Analysis")
                                
                                if detailed.get('summary'):
                                    st.write("**Summary:**")
                                    for point in detailed['summary']:
                                        st.write(f"• {point}")
                                
                                if detailed.get('strengths'):
                                    st.write("**Strengths:**")
                                    for strength in detailed['strengths']:
                                        st.write(f"✅ {strength}")
                                
                                if detailed.get('weaknesses'):
                                    st.write("**Areas for Improvement:**")
                                    for weakness in detailed['weaknesses']:
                                        st.write(f"⚠️ {weakness}")
                
                with tab2:
                    st.subheader("📄 Resume Content Preview")
                    if result.get('extracted_text_path') and os.path.exists(result['extracted_text_path']):
                        with open(result['extracted_text_path'], 'r', encoding='utf-8') as f:
                            resume_text = f.read()
                        st.text_area("Resume Text", resume_text, height=400, disabled=True, label_visibility="collapsed")
                    else:
                        st.warning("Resume text not available for preview")
                
                with tab3:
                    st.subheader("📋 Job Description Preview")
                    if st.session_state.jd_text:
                        st.text_area("Job Description Text", st.session_state.jd_text, height=400, disabled=True, label_visibility="collapsed")
                    else:
                        st.warning("Job description not available for preview")
            
        # Export buttons
        st.markdown("---")
        st.subheader("📥 Export Results")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"resume_scores_{int(time.time())}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"resume_scores_{int(time.time())}.json",
                mime="application/json"
            )
    else:
        st.info("Upload job descriptions and resumes to see analysis results.")

if __name__ == "__main__":
    main()
