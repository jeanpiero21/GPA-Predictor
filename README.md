# Predictor de Rendimiento Académico (MOLps) — README

> Herramienta que estima el **GPA final** de estudiantes de primer semestre para identificar tempranamente a quienes necesitan apoyo académico. Esta versión usa la base de datos provista (columnas: 'StudentID, Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering, GPA, GradeClass'). La app se desplegará con **Streamlit** y los modelos se guardarán/recargarán usando archivos '*.pkl' (pickle).

---

## 1. Objetivo del proyecto
Construir un modelo predictivo que:
- Prediga el **GPA** (variable target: 'GPA') con alta precisión.
- Evite introducir o amplificar sesgos sociales (por ejemplo de género o etnia).
- Provea una interfaz motivacional y accionable para estudiantes y coordinadores.
- Sea reproducible y desplegable con Streamlit.

---

## 2. Consideraciones sobre variables (qué excluir y por qué)

### 2.1 Variables **no informativas** / eliminar
- 'StudentID' — identificador único; **no aporta información predictiva** y su uso puede violar privacidad. **Se ha Eliminado**.

### 2.2 Variables potencialmente **sesgadoras / sensibles** (**no se han usado como features de entrada**)
Estas variables pueden introducir discriminación histórica o amplificar desigualdades.
- 'Gender' — puede llevar a decisiones sesgadas por género.
- 'Ethnicity' — alto riesgo de discriminación étnica.
- 'ParentalEducation' — proxy de estatus socioeconómico; puede reforzar desigualdades.
- 'ParentalSupport' — puede reflejar recursos familiares (sesgo socioeconómico).

**Estrategias alternativas**:
- **Eliminar** estas features del modelo final si el objetivo es evitar sesgo.
- Si por análisis las necesitas (p. ej. para aumentar precisión y luego mitigar sesgo), **mantenerlas fuera del modelo usado en decisiones automáticas** o usarlas solo para análisis post-hoc / explicaciones agregadas.
- Si decides modelarlas, aplicar **métodos de equidad**: re-ponderación, constraints de fairness, o post-processing (p. ej. equalized odds), y reportar métricas de equidad.

### 2.3 Variables recomendadas para mantener (informativas)
- 'Age' — puede correlacionar con madurez/experiencia.
- 'StudyTimeWeekly' — directamente relacionado con rendimiento.
- 'Absences' — impacto negativo esperado en GPA.
- 'Tutoring' — indica intervención educativa.
- 'Extracurricular', 'Sports', 'Music', 'Volunteering' — actividades que pueden correlacionar positiva o negativamente con rendimiento.

### 2.4 Variable especial
- 'GradeClass' — puede ser una versión categórica/agrupada del GPA (p. ej. clases 0..4). **No se usa como feature si por se derivada del GPA** (riesgo de fuga de información). Este clasificacion se puede usar en la salida del modelo de machine learning.

---