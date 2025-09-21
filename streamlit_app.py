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
                                "resume_file": resume_file.name,
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
                        [
                            {
                                "resume_file": r["resume_file"],
                                "text": r.get("text", ""),
                                "missing_skills": r.get("missing_skills", [])
                            } for r in resumes_data
                        ],
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
                    # Debug: print resume_file values to terminal
                    print("[DEBUG] Resume filenames in results:", [r.get("resume_file") for r in final_results])
                    
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
                # Create columns for the card (no summary at top)
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**📄 {result['resume_file']}**")
                with col2:
                    verdict_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
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
                # Show missing skills only
                col_skills, col_report = st.columns([1,2])
                with col_skills:
                    if result['missing_skills']:
                        with st.expander("❌ Missing Skills", expanded=False):
                            for skill in result['missing_skills']:
                                st.write(f"• {skill}")
                    else:
                        st.success("✅ All required skills found!")
                # Remove recruiter report from card; show only in detailed analysis tab
        
        # Detailed view for selected resume
        if len(st.session_state.results) > 0:
            st.markdown("---")
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
                        # Add detailed feedback if available
                        if 'detailed_feedback' in result:
                            detailed = result['detailed_feedback']
                            import json
                            # Always parse detailed as dict if it's a string
                            if isinstance(detailed, str):
                                try:
                                    detailed_json = json.loads(detailed)
                                    if isinstance(detailed_json, dict):
                                        detailed = detailed_json
                                    else:
                                        detailed = {}
                                except Exception:
                                    detailed = {}
                            # Now safe to use .get()
                            summary = detailed.get('summary', [])
                            strengths = detailed.get('strengths', [])
                            weaknesses = detailed.get('weaknesses', [])
                            courses = detailed.get('recommended_courses', [])
                            # If any field is a stringified JSON, parse it
                            def parse_json_list(val):
                                if isinstance(val, str):
                                    try:
                                        parsed = json.loads(val)
                                        if isinstance(parsed, list):
                                            return parsed
                                        elif isinstance(parsed, dict) and 'summary' in parsed:
                                            return parsed['summary']
                                    except Exception:
                                        return []
                                return val
                            summary = parse_json_list(summary)
                            strengths = parse_json_list(strengths)
                            weaknesses = parse_json_list(weaknesses)
                            courses = parse_json_list(courses)
                            if summary or strengths or weaknesses or courses:
                                st.subheader("📝 Detailed Analysis")
                                if summary and isinstance(summary, list):
                                    st.write("**Summary:**")
                                    for point in summary:
                                        if isinstance(point, str):
                                            st.write(f"• {point}")
                                strengths_clean = [s for s in strengths if s and not str(s).strip().startswith('LLM-generated')]
                                if strengths_clean:
                                    st.write("**Strengths:**")
                                    for strength in strengths_clean:
                                        if isinstance(strength, str):
                                            st.write(f"✅ {strength}")
                                if weaknesses and isinstance(weaknesses, list):
                                    st.write("**Areas for Improvement:**")
                                    for weakness in weaknesses:
                                        if isinstance(weakness, str):
                                            st.write(f"⚠️ {weakness}")
                                if courses and isinstance(courses, list):
                                    st.write("**Suggested Courses:**")
                                    for course in courses:
                                        if isinstance(course, dict):
                                            title = course.get('title') or course.get('name') or str(course)
                                            url = course.get('url', '')
                                            if url:
                                                st.write(f"🔗 [{title}]({url})")
                                            else:
                                                st.write(f"🔗 {title}")
                                        else:
                                            st.write(f"🔗 {course}")
                
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
            
        # PDF Export button only
        st.markdown("---")
        st.subheader("📥 Export Results")
        from fpdf import FPDF
        import textwrap

        def generate_pdf(results, jd_text: str = "", selected_index: int | None = None) -> bytes:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Prepare which results to include
            if selected_index is not None and 0 <= selected_index < len(results):
                to_export = [results[selected_index]]
            else:
                to_export = results

            for idx, r in enumerate(to_export):
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                title = f"Resume {idx+1}: {r.get('resume_file', '')}"
                pdf.cell(0, 10, title, ln=True)
                pdf.ln(2)

                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 8, f"Final Score: {r.get('final_score', '')}")
                pdf.multi_cell(0, 8, f"Verdict: {r.get('verdict', '')}")

                missing_skills = r.get('missing_skills', [])
                if missing_skills:
                    pdf.multi_cell(0, 8, "Missing Skills:")
                    for skill in missing_skills:
                        pdf.multi_cell(0, 8, f"- {skill}")

                if r.get('grok_feedback'):
                    pdf.set_font("Arial", "B", 11)
                    pdf.multi_cell(0, 8, f"AI Feedback: {r.get('grok_feedback')}")
                    pdf.set_font("Arial", size=11)

                detailed = r.get('detailed_feedback', {}) or {}
                if detailed:
                    summary = detailed.get('summary', [])
                    if summary:
                        pdf.multi_cell(0, 8, "Summary:")
                        for point in summary:
                            pdf.multi_cell(0, 8, f"- {point}")

                    strengths = [s for s in detailed.get('strengths', []) if s and not str(s).strip().startswith('LLM-generated')]
                    if strengths:
                        pdf.multi_cell(0, 8, "Strengths:")
                        for s in strengths:
                            pdf.multi_cell(0, 8, f"+ {s}")

                    weaknesses = detailed.get('weaknesses', [])
                    if weaknesses:
                        pdf.multi_cell(0, 8, "Areas for Improvement:")
                        for w in weaknesses:
                            pdf.multi_cell(0, 8, f"! {w}")

                    courses = detailed.get('recommended_courses', [])
                    if courses:
                        pdf.multi_cell(0, 8, "Suggested Courses:")
                        for course in courses:
                            if isinstance(course, dict):
                                name = course.get('name') or course.get('title') or str(course)
                                url = course.get('url', '')
                                if url and name:
                                    pdf.multi_cell(0, 8, f"* {name} ({url})")
                                elif name:
                                    pdf.multi_cell(0, 8, f"* {name}")
                                elif url:
                                    pdf.multi_cell(0, 8, f"* {url}")
                                else:
                                    pdf.multi_cell(0, 8, f"* {course}")
                            elif isinstance(course, str):
                                pdf.multi_cell(0, 8, f"* {course}")

                # Include resume extracted text if available
                extracted_path = r.get('extracted_text_path')
                if extracted_path and os.path.exists(extracted_path):
                    try:
                        with open(extracted_path, 'r', encoding='utf-8') as f:
                            resume_text = f.read().strip()
                        if resume_text:
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 12)
                            pdf.cell(0, 8, "Resume Text", ln=True)
                            pdf.set_font("Arial", size=10)
                            for line in resume_text.splitlines():
                                wrapped = textwrap.wrap(line, 90)
                                if not wrapped:
                                    pdf.multi_cell(0, 6, "")
                                for wline in wrapped:
                                    pdf.multi_cell(0, 6, wline)
                    except Exception:
                        pass

                # Include job description text once (for first resume exported)
                if jd_text and (selected_index is not None or idx == 0):
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 8, "Job Description", ln=True)
                    pdf.set_font("Arial", size=10)
                    for line in jd_text.splitlines():
                        wrapped = textwrap.wrap(line, 90)
                        if not wrapped:
                            pdf.multi_cell(0, 6, "")
                        for wline in wrapped:
                            pdf.multi_cell(0, 6, wline)

            # Return bytes encoded for streamlit download
            raw = pdf.output(dest='S')
            if isinstance(raw, str):
                raw = raw.encode('latin-1')
            return raw

        selected_idx = st.session_state.get('selected_resume') if st.session_state.get('selected_resume') is not None else None
        pdf_bytes = generate_pdf(st.session_state.results, jd_text=str(st.session_state.get('jd_text') or ''), selected_index=selected_idx)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"resume_scores_{int(time.time())}.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Upload job descriptions and resumes to see analysis results.")

if __name__ == "__main__":
    main()
