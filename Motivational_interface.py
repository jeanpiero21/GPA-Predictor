import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Rendimiento Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Aplicación iniciada")

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        with open('WeightBestModel.pkl', 'rb') as file:
            model = joblib.load(file)
        logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        st.error("❌ Error al cargar el modelo. Contacte al administrador.")
        return None

# Cargar frases motivacionales
def get_motivational_quotes(gpa):
    quotes = {
        "low": [
            "«El éxito no se mide por lo que logras, sino por los obstáculos que superas.» - Booker T. Washington",
            "«Nunca es demasiado tarde para ser lo que podrías haber sido.» - George Eliot",
            "«El fracaso es simplemente la oportunidad de comenzar de nuevo, esta vez de manera más inteligente.» - Henry Ford",
            "«La educación es el pasaporte hacia el futuro, el mañana pertenece a aquellos que se preparan para él en el hoy.» - Malcolm X",
            "«No midas tu éxito por lo lejos que has llegado, sino por la distancia que has recorrido desde donde empezaste.»"
        ],
        "medium": [
            "«La constancia es el camino al éxito.» - Charles Chaplin",
            "«El talento gana partidos, pero el trabajo en equipo y la inteligencia ganan campeonatos.» - Michael Jordan",
            "«El conocimiento es poder. La información es liberadora. La educación es la premisa del progreso.» - Kofi Annan",
            "«Cada maestro fue primero un estudiante. Cada experto fue primero un principiante.»",
            "«El progreso es imposible sin cambio, y aquellos que no pueden cambiar sus mentes no pueden cambiar nada.» - George Bernard Shaw"
        ],
        "high": [
            "«La excelencia no es un acto, sino un hábito.» - Aristóteles",
            "«Cuanto más sudas en el entrenamiento, menos sangras en la batalla.» - Proverbio",
            "«La mente es como un paracaídas, solo funciona si se abre.» - Albert Einstein",
            "«El éxito es la suma de pequeños esfuerzos repetidos día tras día.» - Robert Collier",
            "«La educación no es la preparación para la vida; la educación es la vida misma.» - John Dewey"
        ],
        "excellent": [
            "«El único límite para nuestra realización de mañana serán nuestras dudas de hoy.» - Franklin D. Roosevelt",
            "«La función de la educación es enseñar a pensar intensamente y críticamente. Inteligencia más carácter: esa es la meta de la verdadera educación.» - Martin Luther King Jr.",
            "«No busques ser exitoso, busca ser valioso y el éxito llegará.» - Albert Einstein",
            "«Los grandes logros de cualquier época generalmente fueron las posibilidades de la imaginación de la época anterior.»",
            "«El futuro pertenece a aquellos que creen en la belleza de sus sueños.» - Eleanor Roosevelt"
        ]
    }
    
    if gpa < 2.0:
        return np.random.choice(quotes["low"])
    elif gpa < 3.0:
        return np.random.choice(quotes["medium"])
    elif gpa < 3.7:
        return np.random.choice(quotes["high"])
    else:
        return np.random.choice(quotes["excellent"])

# Obtener recomendaciones según GPA
def get_recommendations(gpa, student_data):
    recommendations = []
    
    if gpa < 2.0:  # Alto riesgo
        recommendations.append("🔴 **Intervención urgente necesaria**")
        recommendations.append("• Programa de tutorías intensivas (3+ sesiones semanales)")
        recommendations.append("• Reunión con el coordinador académico esta semana")
        recommendations.append("• Revisión del plan de estudio y técnicas de aprendizaje")
        recommendations.append("• Evaluación de posibles problemas externos que afecten el rendimiento")
        
        if student_data['StudyTimeWeekly'] < 10:
            recommendations.append("• Incrementar tiempo de estudio a mínimo 15 horas semanales")
        if student_data['Absences'] > 10:
            recommendations.append("• Control de asistencia y plan para reducir faltas")
        if student_data['Tutoring'] == 0:
            recommendations.append("• Inscribirse inmediatamente en el programa de tutorías")
            
    elif gpa < 3.0:  # Riesgo moderado
        recommendations.append("🟡 **Intervención preventiva recomendada**")
        recommendations.append("• Participación en tutorías (2 sesiones semanales)")
        recommendations.append("• Talleres de técnicas de estudio y gestión del tiempo")
        recommendations.append("• Revisión de materias con mayor dificultad")
        
        if student_data['StudyTimeWeekly'] < 15:
            recommendations.append("• Aumentar tiempo de estudio a 15-20 horas semanales")
        if student_data['Extracurricular'] == 0 and student_data['Sports'] == 0 and student_data['Music'] == 0 and student_data['Volunteering'] == 0:
            recommendations.append("• Considerar participar en alguna actividad extracurricular para mejorar el equilibrio")
            
    elif gpa < 3.7:  # Bajo riesgo
        recommendations.append("🟢 **Rendimiento satisfactorio**")
        recommendations.append("• Mantener buenos hábitos de estudio")
        recommendations.append("• Identificar áreas de mejora para alcanzar excelencia")
        recommendations.append("• Considerar mentoría para estudiantes con mayor dificultad")
        
        if student_data['StudyTimeWeekly'] > 25:
            recommendations.append("• Evaluar técnicas de estudio para mejorar eficiencia")
            
    else:  # Excelente
        recommendations.append("✅ **Rendimiento sobresaliente**")
        recommendations.append("• Considerar programas de honores o investigación")
        recommendations.append("• Mentoría para otros estudiantes")
        recommendations.append("• Explorar oportunidades de liderazgo académico")
        recommendations.append("• Participar en conferencias o competencias académicas")
    
    return recommendations

# Determinar nivel de riesgo
def get_risk_level(gpa):
    if gpa < 2.0:
        return "🔴 ALTO RIESGO", 4
    elif gpa < 3.0:
        return "🟡 RIESGO MODERADO", 3
    elif gpa < 3.7:
        return "🟢 BAJO RIESGO", 2
    else:
        return "✅ EXCELENTE", 1

# Función para predecir GPA
def predict_gpa(input_data):
    model = load_model()
    if model is None:
        return None
    
    try:
        # Asegurar el orden correcto de las características
        features = ['Age', 'StudyTimeWeekly', 'Absences', 'Tutoring', 
                   'Extracurricular', 'Sports', 'Music', 'Volunteering']
        input_df = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return None

# Interfaz para estudiantes
def student_interface():
    st.header("🎓 Predictor de Rendimiento Académico - Vista Estudiante")
    st.info("¡Hola! Completa la información a continuación para conocer tu predicción de GPA y recibir recomendaciones personalizadas.")
    
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", 15, 25, 18)
            study_time = st.slider("Horas de estudio semanales", 0.0, 40.0, 15.0, 0.5)
            absences = st.slider("Número de ausencias", 0, 30, 5)
            tutoring = st.selectbox("¿Participas en tutorías?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            
        with col2:
            extracurricular = st.selectbox("¿Actividades extracurriculares?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            sports = st.selectbox("¿Practicas deportes?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            music = st.selectbox("¿Practicas música?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            volunteering = st.selectbox("¿Participas en voluntariado?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        
        submitted = st.form_submit_button("Predecir mi GPA", type="primary")
    
    if submitted:
        # Preparar datos para predicción
        input_data = [age, study_time, absences, tutoring, extracurricular, sports, music, volunteering]
        student_data = {
            'Age': age,
            'StudyTimeWeekly': study_time,
            'Absences': absences,
            'Tutoring': tutoring,
            'Extracurricular': extracurricular,
            'Sports': sports,
            'Music': music,
            'Volunteering': volunteering
        }
        
        # Realizar predicción
        with st.spinner("Analizando tu información..."):
            gpa = predict_gpa(input_data)
        
        if gpa is not None:
            # Registrar la predicción
            logger.info(f"Predicción estudiante - GPA: {gpa:.2f}, Datos: {student_data}")
            
            # Mostrar resultados
            st.success("¡Análisis completado!")
            
            # Mostrar GPA y nivel de riesgo
            risk_level, risk_code = get_risk_level(gpa)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("GPA Predicho", f"{gpa:.2f}")
                st.info(risk_level)
            
            with col2:
                # Visualización simple del GPA
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh([0], [gpa], color=['red' if gpa < 2.0 else 'orange' if gpa < 3.0 else 'green' if gpa < 3.7 else 'blue'])
                ax.set_xlim(0, 4.0)
                ax.set_xlabel('GPA')
                ax.set_yticks([])
                ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=3.0, color='orange', linestyle='--', alpha=0.5)
                ax.axvline(x=3.7, color='green', linestyle='--', alpha=0.5)
                ax.set_title('Tu GPA en escala de 4.0')
                st.pyplot(fig)
            
            # Mostrar frase motivacional
            quote = get_motivational_quotes(gpa)
            st.info(f"💡 **Frase motivacional:** {quote}")
            
            # Mostrar recomendaciones
            st.subheader("📋 Recomendaciones personalizadas")
            recommendations = get_recommendations(gpa, student_data)
            for rec in recommendations:
                st.write(rec)
            
            # Guardar en registro de estudiantes (simulado)
            try:
                student_log = {
                    'timestamp': datetime.now(),
                    'age': age,
                    'study_time': study_time,
                    'absences': absences,
                    'tutoring': tutoring,
                    'extracurricular': extracurricular,
                    'sports': sports,
                    'music': music,
                    'volunteering': volunteering,
                    'gpa_predicted': gpa,
                    'risk_level': risk_code
                }
                
                # Aquí iría el código para guardar en base de datos
                # Por ahora solo log
                logger.info(f"Registro estudiante guardado: {student_log}")
            except Exception as e:
                logger.error(f"Error guardando registro: {str(e)}")
        else:
            st.error("Error al calcular la predicción. Intenta nuevamente.")

# Interfaz para coordinadores - Opción 1: Ingresar datos manualmente
def coordinator_manual_input():
    st.header("👨‍🏫 Vista Coordinador - Análisis Individual")
    st.info("Ingresa los datos del estudiante para analizar su situación académica.")
    
    with st.form("coordinator_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input("ID del Estudiante", value="")
            age = st.slider("Edad", 15, 25, 18)
            study_time = st.slider("Horas de estudio semanales", 0.0, 40.0, 15.0, 0.5)
            absences = st.slider("Número de ausencias", 0, 30, 5)
            tutoring = st.selectbox("¿Participa en tutorías?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            
        with col2:
            extracurricular = st.selectbox("¿Actividades extracurriculares?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            sports = st.selectbox("¿Practica deportes?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            music = st.selectbox("¿Practica música?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
            volunteering = st.selectbox("¿Participa en voluntariado?", options=[("Sí", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        
        submitted = st.form_submit_button("Analizar Estudiante", type="primary")
    
    if submitted:
        input_data = [age, study_time, absences, tutoring, extracurricular, sports, music, volunteering]
        student_data = {
            'StudentID': student_id,
            'Age': age,
            'StudyTimeWeekly': study_time,
            'Absences': absences,
            'Tutoring': tutoring,
            'Extracurricular': extracurricular,
            'Sports': sports,
            'Music': music,
            'Volunteering': volunteering
        }
        
        with st.spinner("Analizando información del estudiante..."):
            gpa = predict_gpa(input_data)
        
        if gpa is not None:
            # Registrar la predicción
            logger.info(f"Predicción coordinador - Estudiante: {student_id}, GPA: {gpa:.2f}")
            
            # Mostrar resultados
            st.success("Análisis completado")
            
            risk_level, risk_code = get_risk_level(gpa)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("GPA Predicho", f"{gpa:.2f}")
            
            with col2:
                st.metric("Nivel de Riesgo", risk_level)
            
            with col3:
                st.metric("Intervención", "Requerida" if risk_code >= 3 else "Preventiva" if risk_code == 2 else "No requerida")
            
            # Mostrar recomendaciones detalladas
            st.subheader("📋 Plan de Intervención")
            recommendations = get_recommendations(gpa, student_data)
            for rec in recommendations:
                st.write(rec)
            
            # Guardar en registro de coordinadores
            try:
                coord_log = {
                    'timestamp': datetime.now(),
                    'coordinator_action': 'manual_analysis',
                    'student_id': student_id,
                    'gpa_predicted': gpa,
                    'risk_level': risk_code
                }
                logger.info(f"Registro coordinador guardado: {coord_log}")
            except Exception as e:
                logger.error(f"Error guardando registro coordinador: {str(e)}")
        else:
            st.error("Error al calcular la predicción. Intenta nuevamente.")

# Interfaz para coordinadores - Opción 2: Ver estudiantes en riesgo
def coordinator_risk_list():
    st.header("👨‍🏫 Vista Coordinador - Lista de Estudiantes en Riesgo")
    st.info("Visualiza los estudiantes identificados con mayor necesidad de intervención.")
    
    # En una implementación real, esto vendría de una base de datos
    # Por ahora simulamos algunos datos basados en registros del log
    
    try:
        # Leer el archivo de log para extraer predicciones anteriores
        with open('logs/app.log', 'r') as f:
            log_lines = f.readlines()
        
        student_entries = []
        for line in log_lines:
            if 'Predicción estudiante - GPA:' in line:
                parts = line.split('Datos:')
                if len(parts) > 1:
                    # Extraer información básica (simplificado)
                    gpa_part = parts[0].split('GPA: ')[1].split(',')[0]
                    try:
                        gpa = float(gpa_part)
                        # Solo considerar estudiantes en riesgo
                        if gpa < 3.0:
                            student_entries.append({'GPA': gpa, 'Datos': parts[1].strip()})
                    except:
                        continue
        
        if student_entries:
            # Ordenar por GPA (menor primero)
            student_entries.sort(key=lambda x: x['GPA'])
            
            st.subheader(f"Estudiantes identificados con riesgo académico: {len(student_entries)}")
            
            for i, entry in enumerate(student_entries[:10]):  # Mostrar máximo 10
                risk_level, risk_code = get_risk_level(entry['GPA'])
                
                with st.expander(f"Estudiante {i+1} - GPA: {entry['GPA']:.2f} - {risk_level}"):
                    st.write(f"**Datos:** {entry['Datos']}")
                    st.write(f"**Recomendación:** {'Intervención urgente' if risk_code == 4 else 'Intervención preventiva'}")
                    
                    # Recomendaciones genéricas basadas en nivel de riesgo
                    if risk_code == 4:
                        st.write("**Acciones inmediatas:**")
                        st.write("- Contactar al estudiante dentro de 24 horas")
                        st.write("- Programar evaluación psicopedagógica")
                        st.write("- Establecer plan de mejora académica")
                    else:
                        st.write("**Acciones recomendadas:**")
                        st.write("- Invitar al estudiante a programa de tutorías")
                        st.write("- Monitorear asistencia y participación")
                        st.write("- Revisar carga académica")
        else:
            st.info("No se encontraron estudiantes en riesgo en los registros actuales.")
            
    except Exception as e:
        st.error(f"Error al leer los registros: {str(e)}")
        logger.error(f"Error procesando lista de riesgo: {str(e)}")

# Interfaz principal
def main():
    # Sidebar con selección de modo
    st.sidebar.title("🎓 Predictor de Rendimiento Académico")
    
    app_mode = st.sidebar.radio("Selecciona tu modo:",
                                ["Estudiante", "Coordinador Académico"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    Esta herramienta predictiva estima el GPA final de estudiantes universitarios 
    de primer semestre para identificar tempranamente a quienes necesitan apoyo académico.
    """)
    
    # Mostrar estadísticas en sidebar
    try:
        with open('logs/app.log', 'r') as f:
            log_content = f.read()
        
        student_predictions = log_content.count('Predicción estudiante - GPA:')
        coordinator_actions = log_content.count('Predicción coordinador - Estudiante:')
        
        st.sidebar.markdown("### 📊 Estadísticas")
        st.sidebar.write(f"Predicciones estudiantiles: **{student_predictions}**")
        st.sidebar.write(f"Acciones de coordinadores: **{coordinator_actions}**")
        
    except:
        st.sidebar.write("No hay estadísticas disponibles")
    
    # Contenido principal según selección
    if app_mode == "Estudiante":
        student_interface()
    else:
        coordinator_mode = st.sidebar.radio("Tipo de análisis:",
                                           ["Análisis Individual", "Lista de Estudiantes en Riesgo"])
        
        if coordinator_mode == "Análisis Individual":
            coordinator_manual_input()
        else:
            coordinator_risk_list()

if __name__ == "__main__":
    main()