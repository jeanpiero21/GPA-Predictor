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


# Agregar esto al inicio del archivo, después de los imports
if 'stats_updated' not in st.session_state:
    st.session_state.stats_updated = False
if 'student_count' not in st.session_state:
    st.session_state.student_count = 0
if 'coordinator_count' not in st.session_state:
    st.session_state.coordinator_count = 0

# Función para actualizar estadísticas
def update_stats():
    try:
        if os.path.exists('logs/app.log'):
            with open('logs/app.log', 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            st.session_state.student_count = log_content.count('Predicción estudiante - GPA:')
            st.session_state.coordinator_count = log_content.count('Predicción coordinador - Estudiante:')
            st.session_state.stats_updated = True
    except:
        st.session_state.student_count = 0
        st.session_state.coordinator_count = 0

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

# Especificar encoding UTF-8 explícitamente
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),  # ¡Aquí está el fix!
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
# Interfaz para coordinadores - Opción 2: Ver estudiantes en riesgo
def coordinator_risk_list():
    st.header("👨‍🏫 Vista Coordinador - Lista de Estudiantes en Riesgo")
    st.info("Visualiza los estudiantes identificados con mayor necesidad de intervención.")
    
    # En una implementación real, esto vendría de una base de datos
    # Por ahora simulamos algunos datos basados en registros del log
    
    try:
        # Leer el archivo de log con encoding UTF-8
        with open('logs/app.log', 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        student_entries = []
        for line in log_lines:
            if 'Predicción estudiante - GPA:' in line:
                try:
                    # Extraer GPA usando un método más robusto
                    gpa_match = line.split('GPA: ')[1].split(',')[0].strip()
                    gpa = float(gpa_match)
                    
                    # Extraer datos del estudiante
                    if 'Datos:' in line:
                        data_part = line.split('Datos:')[1].strip()
                        # Solo considerar estudiantes en riesgo
                        if gpa < 3.0:
                            student_entries.append({
                                'GPA': gpa, 
                                'Datos': data_part,
                                'Línea': line.strip()  # Para debugging
                            })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line: {line.strip()} - {e}")
                    continue
        
        if student_entries:
            # Ordenar por GPA (menor primero)
            student_entries.sort(key=lambda x: x['GPA'])
            
            st.subheader(f"🎯 Estudiantes identificados con riesgo académico: {len(student_entries)}")
            
            # Mostrar resumen estadístico
            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk = sum(1 for entry in student_entries if entry['GPA'] < 2.0)
                st.metric("Alto Riesgo", high_risk)
            with col2:
                medium_risk = sum(1 for entry in student_entries if 2.0 <= entry['GPA'] < 3.0)
                st.metric("Riesgo Moderado", medium_risk)
            with col3:
                avg_gpa = sum(entry['GPA'] for entry in student_entries) / len(student_entries)
                st.metric("GPA Promedio", f"{avg_gpa:.2f}")
            
            # Mostrar detalles de cada estudiante
            for i, entry in enumerate(student_entries[:15]):  # Mostrar máximo 15
                risk_level, risk_code = get_risk_level(entry['GPA'])
                risk_color = "🔴" if risk_code == 4 else "🟡" if risk_code == 3 else "🟢"
                
                with st.expander(f"{risk_color} Estudiante {i+1} - GPA: {entry['GPA']:.2f} - {risk_level}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Información del Estudiante:**")
                        st.code(entry['Datos'], language='json')
                    
                    with col2:
                        st.write("**📋 Recomendaciones:**")
                        
                        if risk_code == 4:
                            st.error("**Intervención Urgente Requerida**")
                            st.write("• Contactar dentro de 24 horas")
                            st.write("• Evaluación psicopedagógica inmediata")
                            st.write("• Plan de mejora académica intensivo")
                            st.write("• Reunión con coordinador esta semana")
                            
                        elif risk_code == 3:
                            st.warning("**Intervención Preventiva**")
                            st.write("• Invitar a programa de tutorías")
                            st.write("• Monitorear asistencia semanal")
                            st.write("• Revisar técnicas de estudio")
                            st.write("• Establecer metas académicas")
                        
                        # Botón para acción
                        if st.button(f"📞 Contactar Estudiante {i+1}", key=f"contact_{i}"):
                            st.success(f"Acción de contacto iniciada para estudiante {i+1}")
            
            # Opción para exportar la lista
            if st.button("📤 Exportar Lista de Riesgo"):
                # Crear DataFrame para exportación
                df_export = pd.DataFrame(student_entries)
                df_export = df_export[['GPA', 'Datos']]  # Solo columnas relevantes
                
                # Convertir a CSV
                csv = df_export.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="estudiantes_riesgo.csv",
                    mime="text/csv"
                )
                
        else:
            st.success("✅ No se encontraron estudiantes en riesgo en los registros actuales.")
            st.info("""
            **Posibles razones:**
            - Todos los estudiantes tienen buen rendimiento (GPA ≥ 3.0)
            - No hay predicciones registradas aún
            - Los registros pueden estar en un formato diferente
            """)
            
    except FileNotFoundError:
        st.warning("📝 No se encontró el archivo de registros. Realiza algunas predicciones primero.")
        st.info("La lista de estudiantes en riesgo se generará automáticamente una vez que haya predicciones.")
        
    except Exception as e:
        st.error(f"❌ Error al procesar los registros: {str(e)}")
        logger.error(f"Error procesando lista de riesgo: {str(e)}")
        
        # Información de debugging
        with st.expander("🔧 Información de Debugging"):
            st.write("**Error details:**", str(e))
            st.write("**Solución:** Asegúrate de que el archivo logs/app.log existe y tiene el formato correcto.")
            
            # Intentar listar archivos en directorio logs
            try:
                if os.path.exists('logs'):
                    files = os.listdir('logs')
                    st.write("**Archivos en directorio logs:**", files)
                else:
                    st.write("El directorio 'logs' no existe")
            except:
                st.write("No se pudo acceder al directorio logs")
# Interfaz principal
def main():
    # Actualizar estadísticas al iniciar
    update_stats()
    
    # Sidebar con selección de modo
    st.sidebar.title("🎓 Predictor de Rendimiento Académico")
    
    app_mode = st.sidebar.radio("Selecciona tu modo:",
                                ["Estudiante", "Coordinador Académico"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    Esta herramienta predictiva estima el GPA final de estudiantes universitarios 
    de primer semestre para identificar tempranamente a quienes necesitan apoyo académico.
    """)
    
    # Mostrar estadísticas en sidebar (ACTUALIZADO)
    st.sidebar.markdown("### 📊 Estadísticas")
    st.sidebar.write(f"Predicciones estudiantiles: **{st.session_state.student_count}**")
    st.sidebar.write(f"Acciones de coordinadores: **{st.session_state.coordinator_count}**")
    
    # Botón para actualizar manualmente
    if st.sidebar.button("🔄 Actualizar estadísticas"):
        update_stats()
        st.sidebar.success("Estadísticas actualizadas")
    
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