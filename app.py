import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

@st.cache_resource
def load_and_train_models():
    df = sns.load_dataset("titanic")
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']
    X = df[features].copy()
    y = df['survived'].astype('int8')

    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric_features),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel='rbf', random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    pipelines = {}
    predictions = {}
    metrics = []
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
        predictions[name] = pipeline.predict(X_test)
        metrics.append({'Model': name, 'Accuracy': accuracy_score(y_test, predictions[name])})

    metrics_df = pd.DataFrame(metrics).sort_values('Accuracy', ascending=False)
    return pipelines, df, X_test, y_test, predictions, metrics_df

try:
    models, raw_df, X_test, y_test, test_predictions, metrics_df = load_and_train_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


def show_passenger_summary(data):
    """Display dataset size and survival counts by sex."""
    total_passengers = len(data)
    male_count = (data['sex'] == 'male').sum()
    female_count = (data['sex'] == 'female').sum()
    summary = (
        data.assign(Status=data['survived'].map({1: 'Survived', 0: 'Did not survive'}))
        .groupby(['sex', 'Status'])
        .size()
        .unstack(fill_value=0)
        .reindex(index=['female', 'male'], columns=['Survived', 'Did not survive'], fill_value=0)
    )

    st.info(
        f"The Titanic dataset contains **{total_passengers:,} passengers**: "
        f"**{female_count:,} females** and **{male_count:,} males**. "
        "The table below shows how many passengers in each group survived or did not survive."
    )
    st.dataframe(summary, use_container_width=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Evaluation", "Prediction"])

if page == "Home":
    st.title("🚢 Titanic Survival Prediction App")
    st.markdown("""
    Welcome to the Titanic Survival Prediction app! 
    
    This application uses various Machine Learning models to predict whether a passenger would have survived the Titanic disaster based on features like age, fare, and passenger class.
    
    ### Models Included:
    - **Logistic Regression**
    - **K-Nearest Neighbors (KNN)**
    - **Decision Tree**
    - **Random Forest**
    - **Support Vector Machine (SVM)**
    - **Naive Bayes**
    - **Gradient Boosting**
    
    👈 Use the sidebar to explore the dataset or make your own predictions!
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True, caption="RMS Titanic")

elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    
    st.write("### Dataset Preview")
    st.dataframe(raw_df.head(15))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Survival Rate by Gender")
        st.bar_chart(raw_df.groupby('sex')['survived'].mean() * 100)
    
    with col2:
        st.write("### Survival Rate by Passenger Class")
        st.bar_chart(raw_df.groupby('class')['survived'].mean() * 100)

elif page == "Model Evaluation":
    st.title("📈 Model Charts and Comparisons")
    st.write("All models use the same stratified test set and the same leakage-safe preprocessing pipeline.")
    st.subheader("Passenger Survival Summary")
    show_passenger_summary(raw_df)

    st.subheader("Accuracy by Model")
    st.bar_chart(metrics_df.set_index('Model')['Accuracy'])
    st.dataframe(metrics_df.style.format({'Accuracy': '{:.1%}'}), use_container_width=True)

    st.subheader("Confusion Matrices")
    chart_columns = st.columns(2)
    for index, name in enumerate(metrics_df['Model']):
        matrix = confusion_matrix(y_test, test_predictions[name])
        figure, axis = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axis,
            xticklabels=['Did not survive', 'Survived'],
            yticklabels=['Did not survive', 'Survived'],
        )
        axis.set_title(name)
        axis.set_xlabel('Predicted')
        axis.set_ylabel('Actual')
        with chart_columns[index % 2]:
            st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    st.subheader("Predicted Survival Rate by Sex")
    sex_chart = X_test[['sex']].copy()
    for name, prediction in test_predictions.items():
        sex_chart[name] = prediction
    st.bar_chart(sex_chart.groupby('sex').mean(numeric_only=True))

elif page == "Prediction":
    st.title("🔮 Make a Prediction")
    st.markdown("Enter passenger details below to see survival predictions from our trained models.")
    st.subheader("Dataset Passenger Summary")
    st.write(
        "These are historical counts from the Titanic dataset. They provide context for the "
        "model predictions below; they do not determine the selected passenger's result."
    )
    show_passenger_summary(raw_df)
    
    st.write("### Passenger Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])
        
    if st.button("Predict Survival", type="primary"):
        input_data = pd.DataFrame([{
            'pclass': pclass, 'sex': sex, 'age': age,
            'sibsp': sibsp, 'parch': parch, 'fare': fare,
            'embarked': embarked,
        }])
        
        st.write("---")
        st.subheader("🤖 Model Predictions")
        
        results = {}
        for name, model in models.items():
            pred = int(model.predict(input_data)[0])
            results[name] = pred

        st.subheader("Prediction Chart")
        prediction_chart = pd.DataFrame({
            'Model': list(results),
            'Survival prediction': list(results.values()),
        }).set_index('Model')
        st.bar_chart(prediction_chart)
            
        # Display results in columns
        cols = st.columns(4)
        for i, (name, pred) in enumerate(results.items()):
            with cols[i % 4]:
                if pred == 1:
                    st.success(f"**{name}**\n\nSurvived")
                else:
                    st.error(f"**{name}**\n\nDid Not Survive")
            
        survival_votes = sum(results.values())
        total_models = len(models)
        
        st.write("---")
        st.subheader("🗳️ Majority Vote Decision")
        
        if survival_votes > total_models / 2:
            st.success(f"### The passenger is predicted to **SURVIVE** ({survival_votes}/{total_models} models agree)")
            st.balloons()
        else:
            st.error(f"### The passenger is predicted to **NOT SURVIVE** ({total_models - survival_votes}/{total_models} models agree)")
