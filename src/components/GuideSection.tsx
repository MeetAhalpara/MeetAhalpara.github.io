import { Card } from "@/components/ui/card";
import { CodeBlock } from "./CodeBlock";
import { Brain, Cpu, Cloud, Settings, Zap, Shield, BarChart3, Rocket } from "lucide-react";

interface Step {
  title: string;
  description: string;
  code?: string;
  icon: React.ReactNode;
  tips?: string[];
}

const aiModelSteps: Step[] = [
  {
    title: "1. Environment Setup",
    description: "Set up your development environment with essential AI/ML libraries",
    icon: <Settings className="w-6 h-6" />,
    code: `# Install essential AI/ML libraries
pip install tensorflow keras torch torchvision
pip install scikit-learn pandas numpy matplotlib
pip install jupyter notebook seaborn plotly

# For advanced models
pip install transformers datasets accelerate
pip install langchain openai anthropic

# Set up virtual environment
python -m venv ai_env
source ai_env/bin/activate  # On Windows: ai_env\\Scripts\\activate

# Verify installation
import tensorflow as tf
import torch
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")`,
    tips: [
      "Always use virtual environments to avoid dependency conflicts",
      "Install CUDA if you have an NVIDIA GPU for faster training",
      "Keep libraries updated but test compatibility first"
    ]
  },
  {
    title: "2. Data Preparation",
    description: "Clean, preprocess, and structure your data for optimal model performance",
    icon: <BarChart3 className="w-6 h-6" />,
    code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and explore data
def load_and_explore_data(file_path):
    """Load data and perform initial exploration"""
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Data types: {df.dtypes.value_counts()}")
    
    return df

# Data cleaning pipeline
def clean_data(df):
    """Clean and preprocess data"""
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

# Feature engineering
def engineer_features(df, target_column):
    """Create and transform features"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders`,
    tips: [
      "Always explore your data first - understand distributions and patterns",
      "Handle missing values appropriately for your specific use case",
      "Feature scaling is crucial for neural networks and distance-based algorithms"
    ]
  },
  {
    title: "3. Model Architecture Design",
    description: "Design neural network architecture suitable for your specific problem",
    icon: <Brain className="w-6 h-6" />,
    code: `import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

# Simple Neural Network for Classification
def create_classification_model(input_shape, num_classes):
    """Create a neural network for classification"""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    return model

# Compile and configure model
def compile_model(model, learning_rate=0.001):
    """Compile model with optimizer and loss function"""
    optimizer = Adam(learning_rate=learning_rate)
    
    if model.layers[-1].units == 1:  # Binary classification
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:  # Multi-class classification
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

# Example usage
# model = create_classification_model(input_shape=20, num_classes=3)
# model = compile_model(model)
# print(model.summary())`,
    tips: [
      "Start with simple architectures and increase complexity gradually",
      "Use batch normalization and dropout to prevent overfitting",
      "Choose activation functions based on your problem type"
    ]
  },
  {
    title: "4. Training Process",
    description: "Train your model with proper validation and monitoring",
    icon: <Zap className="w-6 h-6" />,
    code: `import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the model with callbacks and monitoring"""
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    from sklearn.metrics import classification_report
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    return test_accuracy, test_loss`,
    tips: [
      "Use early stopping to prevent overfitting",
      "Monitor both training and validation metrics",
      "Save the best model weights during training"
    ]
  }
];

const aiAgentSteps: Step[] = [
  {
    title: "1. LangChain Agent Setup",
    description: "Create intelligent agents using LangChain framework",
    icon: <Cpu className="w-6 h-6" />,
    code: `# Install required packages
# pip install langchain openai langchain-openai langchain-community

import os
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Create custom tools for the agent
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information"
    
    def _run(self, query: str) -> str:
        # Implement web search functionality
        return f"Search results for: {query}"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)  # Note: Use safely in production
            return f"The result is: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    def _arun(self, expression: str):
        raise NotImplementedError("This tool does not support async")

# Initialize tools
tools = [
    WebSearchTool(),
    CalculatorTool(),
    Tool(
        name="python_repl",
        description="Execute Python code",
        func=lambda x: f"Executed: {x}",
    )
]

# Create memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Example usage
def run_agent(query):
    """Run the agent with a query"""
    try:
        response = agent.run(query)
        return response
    except Exception as e:
        return f"Agent error: {str(e)}"`,
    tips: [
      "Always validate and sanitize inputs to custom tools",
      "Use conversation memory to maintain context",
      "Set appropriate limits on iterations to prevent infinite loops"
    ]
  },
  {
    title: "2. Advanced Agent with RAG",
    description: "Create agents with Retrieval-Augmented Generation capabilities",
    icon: <Brain className="w-6 h-6" />,
    code: `from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader

class RAGAgent:
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.1
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None
        self.agent = None
    
    def load_documents(self, file_paths):
        """Load and process documents for RAG"""
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            docs = loader.load()
            documents.extend(docs)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return len(texts)
    
    def setup_agent(self):
        """Set up the agent with RAG capabilities"""
        if not self.vectorstore:
            raise ValueError("Load documents first!")
        
        # Create RAG tool
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        tools = [
            Tool(
                name="knowledge_base",
                description="Search the knowledge base for relevant information",
                func=lambda x: self.search_knowledge_base(x)
            ),
            CalculatorTool()
        ]
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
    
    def search_knowledge_base(self, query):
        """Search the vector database"""
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\\n".join([doc.page_content for doc in docs])
    
    def query(self, question):
        """Query the RAG agent"""
        if not self.agent:
            raise ValueError("Set up agent first!")
        
        response = self.agent.run(question)
        return response`,
    tips: [
      "Chunk documents appropriately for your use case",
      "Use semantic similarity search for better retrieval",
      "Consider hybrid search combining keyword and semantic search"
    ]
  },
  {
    title: "3. Multi-Agent Systems",
    description: "Coordinate multiple specialized agents for complex tasks",
    icon: <Settings className="w-6 h-6" />,
    code: `class MultiAgentSystem:
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.1
        )
        self.agents = {}
        self.coordinator = None
    
    def create_specialist_agent(self, name, role, tools):
        """Create a specialized agent for specific tasks"""
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
        
        self.agents[name] = {
            'agent': agent,
            'role': role,
            'tools': tools
        }
        
        return agent
    
    def create_coordinator(self):
        """Create a coordinator agent to manage other agents"""
        
        coordinator_tools = [
            Tool(
                name="delegate_to_data_scientist",
                description="Delegate data science and ML tasks",
                func=lambda x: self.delegate_task("data_scientist", x)
            ),
            Tool(
                name="delegate_to_software_engineer", 
                description="Delegate coding and development tasks",
                func=lambda x: self.delegate_task("software_engineer", x)
            ),
            Tool(
                name="delegate_to_researcher",
                description="Delegate research and analysis tasks", 
                func=lambda x: self.delegate_task("researcher", x)
            )
        ]
        
        self.coordinator = initialize_agent(
            tools=coordinator_tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True
        )
    
    def delegate_task(self, agent_name, task):
        """Delegate a task to a specific agent"""
        if agent_name in self.agents:
            try:
                response = self.agents[agent_name]['agent'].run(task)
                return f"Response from {agent_name}: {response}"
            except Exception as e:
                return f"Error from {agent_name}: {str(e)}"
        else:
            return f"Agent {agent_name} not found"
    
    def process_complex_task(self, task):
        """Process a complex task using multiple agents"""
        if not self.coordinator:
            return "Coordinator not set up"
        
        try:
            response = self.coordinator.run(task)
            return response
        except Exception as e:
            return f"Coordinator error: {str(e)}"`,
    tips: [
      "Design agents with clear, non-overlapping responsibilities",
      "Use a coordinator to manage task delegation intelligently",
      "Implement error handling and fallback mechanisms"
    ]
  }
];

const deploymentSteps: Step[] = [
  {
    title: "1. Model Packaging",
    description: "Package your model for production deployment",
    icon: <Shield className="w-6 h-6" />,
    code: `import joblib
import json
import os
from pathlib import Path

class ModelPackager:
    def __init__(self, model_dir="./model_package"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_tensorflow_model(self, model, model_name, metadata=None):
        """Save TensorFlow model with metadata"""
        
        # Save model
        model_path = self.model_dir / f"{model_name}"
        model.save(str(model_path))
        
        # Save metadata
        if metadata:
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def save_sklearn_model(self, model, scaler, model_name, metadata=None):
        """Save scikit-learn model with preprocessing"""
        
        # Save model
        model_path = self.model_dir / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Save metadata
        if metadata:
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Model and scaler saved to: {self.model_dir}")
        return model_path, scaler_path
    
    def create_requirements_file(self, additional_packages=None):
        """Create requirements.txt for deployment"""
        
        base_requirements = [
            "tensorflow>=2.8.0",
            "scikit-learn>=1.0.0", 
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "joblib>=1.0.0",
            "flask>=2.0.0",
            "gunicorn>=20.0.0"
        ]
        
        if additional_packages:
            base_requirements.extend(additional_packages)
        
        req_path = self.model_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            for req in base_requirements:
                f.write(f"{req}\\n")
        
        print(f"Requirements saved to: {req_path}")
        return req_path

# Example usage
def package_model_for_deployment():
    """Complete model packaging example"""
    
    packager = ModelPackager()
    
    # Model metadata
    metadata = {
        "model_name": "customer_churn_predictor",
        "version": "1.0.0",
        "description": "Predicts customer churn probability",
        "input_features": ["age", "tenure", "monthly_charges", "total_charges"],
        "task_type": "classification",
        "performance_metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88
        }
    }
    
    # Create requirements file
    packager.create_requirements_file([
        "flask-cors>=3.0.0",
        "python-dotenv>=0.19.0"
    ])
    
    return packager.model_dir`,
    tips: [
      "Include all preprocessing steps in your model package",
      "Version your models and track metadata",
      "Create comprehensive API documentation"
    ]
  },
  {
    title: "2. Cloud Deployment",
    description: "Deploy your model to cloud platforms (AWS, GCP, Azure)",
    icon: <Cloud className="w-6 h-6" />,
    code: `# Docker deployment setup
from flask import Flask, request, jsonify
import numpy as np
import joblib
import json

# Create Flask API wrapper
def create_model_api():
    """Create Flask API wrapper for the model"""
    
    app = Flask(__name__)
    
    # Load model and preprocessing objects
    model = joblib.load("model_package/model.pkl")
    scaler = joblib.load("model_package/scaler.pkl")
    
    # Load metadata
    with open("model_package/metadata.json", 'r') as f:
        metadata = json.load(f)

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "model": metadata["model_name"]})

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get input data
            data = request.get_json()
            
            if 'features' not in data:
                return jsonify({"error": "Missing features in request"}), 400
            
            # Preprocess input
            features = np.array(data['features']).reshape(1, -1)
            features = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features)
            confidence = float(max(model.predict_proba(features)[0]))
            
            response = {
                "prediction": int(prediction[0]),
                "confidence": confidence,
                "model_version": metadata.get('version', '1.0.0')
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/model-info', methods=['GET'])
    def model_info():
        return jsonify(metadata)

    return app

# AWS deployment with boto3
import boto3

def deploy_to_aws_lambda():
    """Deploy model to AWS Lambda"""
    
    lambda_client = boto3.client('lambda')
    
    # Create deployment package
    with open('lambda_function.py', 'w') as f:
        f.write('''
import json
import numpy as np
import joblib

# Load model (would be included in deployment package)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        features = np.array(body['features']).reshape(1, -1)
        features = scaler.transform(features)
        
        prediction = model.predict(features)
        confidence = float(max(model.predict_proba(features)[0]))
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction[0]),
                'confidence': confidence
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
        ''')

# Example Docker setup
docker_content = '''
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY model_package/ ./model_package/
COPY app.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
'''

print("Model API and deployment configuration ready!")`,
    tips: [
      "Use containerization for consistent deployments",
      "Implement comprehensive monitoring and logging",
      "Set up auto-scaling based on traffic patterns"
    ]
  },
  {
    title: "3. MLOps Pipeline",
    description: "Create automated ML pipeline with CI/CD and monitoring",
    icon: <Rocket className="w-6 h-6" />,
    code: `import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
import optuna

class MLPipeline:
    def __init__(self, experiment_name="ml_experiment"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
    
    def hyperparameter_tuning(self, X_train, y_train, n_trials=50):
        """Automated hyperparameter tuning with Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            units_1 = trial.suggest_int('units_1', 32, 256)
            units_2 = trial.suggest_int('units_2', 16, 128)
            
            # Build model with suggested hyperparameters
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units_1, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(units_2, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate/2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=20,
                validation_split=0.2,
                verbose=0
            )
            
            # Return validation accuracy
            return max(history.history['val_accuracy'])
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train_with_tracking(self, X_train, y_train, X_val, y_val, hyperparams):
        """Train model with experiment tracking"""
        
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(hyperparams)
            
            # Build and compile model
            model = self.build_model(hyperparams)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=hyperparams['batch_size'],
                verbose=1
            )
            
            # Log metrics
            final_accuracy = max(history.history['val_accuracy'])
            final_loss = min(history.history['val_loss'])
            
            mlflow.log_metric("accuracy", final_accuracy)
            mlflow.log_metric("loss", final_loss)
            
            # Log model
            mlflow.tensorflow.log_model(
                model,
                "model",
                registered_model_name="production_model"
            )
            
            return model, history
    
    def build_model(self, hyperparams):
        """Build model with given hyperparameters"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hyperparams['units_1'], 
                activation='relu'
            ),
            tf.keras.layers.Dropout(hyperparams['dropout_rate']),
            tf.keras.layers.Dense(
                hyperparams['units_2'], 
                activation='relu'
            ),
            tf.keras.layers.Dropout(hyperparams['dropout_rate']/2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hyperparams['learning_rate']
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# GitHub Actions workflow example
github_workflow = '''
name: ML Model CI/CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Train model
      run: |
        python scripts/train_model.py
    
    - name: Deploy model
      run: |
        python scripts/deploy_model.py
'''

print("Complete MLOps pipeline setup ready!")`,
    tips: [
      "Automate hyperparameter tuning for optimal performance",
      "Track all experiments and model versions",
      "Implement automated testing and validation pipelines"
    ]
  }
];

export const GuideSection = () => {
  return (
    <section id="ai-guide" className="py-20 px-4 max-w-7xl mx-auto">
      {/* AI Models Section */}
      <div className="mb-20 animate-slide-up">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Brain className="w-8 h-8 text-ai-primary" />
            <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-ai-primary to-ai-secondary bg-clip-text text-transparent">
              Building AI Models
            </h2>
          </div>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Learn to create powerful AI models from scratch with step-by-step Python implementations
          </p>
        </div>

        <div className="grid gap-8">
          {aiModelSteps.map((step, index) => (
            <Card key={index} className="p-8 backdrop-blur-glass border-muted/20 shadow-neural animate-slide-up" style={{ animationDelay: `${index * 0.2}s` }}>
              <div className="flex items-start gap-6 mb-6">
                <div className="p-3 rounded-lg bg-ai-primary/10 text-ai-primary glow-ai">
                  {step.icon}
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold mb-3 text-foreground">{step.title}</h3>
                  <p className="text-muted-foreground text-lg leading-relaxed">{step.description}</p>
                </div>
              </div>
              
              {step.code && (
                <div className="mb-6">
                  <CodeBlock 
                    code={step.code}
                    language="python"
                    title="Implementation"
                  />
                </div>
              )}
              
              {step.tips && (
                <div className="bg-muted/30 rounded-lg p-6 border border-ai-primary/20">
                  <h4 className="font-semibold text-ai-primary mb-3 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Pro Tips
                  </h4>
                  <ul className="space-y-2">
                    {step.tips.map((tip, tipIndex) => (
                      <li key={tipIndex} className="text-sm text-muted-foreground flex items-start gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-ai-accent mt-2 flex-shrink-0" />
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>

      {/* AI Agents Section */}
      <div className="mb-20 animate-slide-up">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Cpu className="w-8 h-8 text-ai-neural" />
            <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-ai-neural to-ai-accent bg-clip-text text-transparent">
              Creating AI Agents
            </h2>
          </div>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Build intelligent agents that can reason, use tools, and coordinate complex tasks
          </p>
        </div>

        <div className="grid gap-8">
          {aiAgentSteps.map((step, index) => (
            <Card key={index} className="p-8 backdrop-blur-glass border-muted/20 shadow-neural animate-slide-up" style={{ animationDelay: `${index * 0.2}s` }}>
              <div className="flex items-start gap-6 mb-6">
                <div className="p-3 rounded-lg bg-ai-neural/10 text-ai-neural glow-ai">
                  {step.icon}
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold mb-3 text-foreground">{step.title}</h3>
                  <p className="text-muted-foreground text-lg leading-relaxed">{step.description}</p>
                </div>
              </div>
              
              {step.code && (
                <div className="mb-6">
                  <CodeBlock 
                    code={step.code}
                    language="python"
                    title="Implementation"
                  />
                </div>
              )}
              
              {step.tips && (
                <div className="bg-muted/30 rounded-lg p-6 border border-ai-neural/20">
                  <h4 className="font-semibold text-ai-neural mb-3 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Pro Tips
                  </h4>
                  <ul className="space-y-2">
                    {step.tips.map((tip, tipIndex) => (
                      <li key={tipIndex} className="text-sm text-muted-foreground flex items-start gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-ai-code mt-2 flex-shrink-0" />
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>

      {/* Deployment Section */}
      <div className="animate-slide-up">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Cloud className="w-8 h-8 text-ai-accent" />
            <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-ai-accent to-ai-code bg-clip-text text-transparent">
              Production Deployment
            </h2>
          </div>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Deploy and scale your AI models in production with modern MLOps practices
          </p>
        </div>

        <div className="grid gap-8">
          {deploymentSteps.map((step, index) => (
            <Card key={index} className="p-8 backdrop-blur-glass border-muted/20 shadow-neural animate-slide-up" style={{ animationDelay: `${index * 0.2}s` }}>
              <div className="flex items-start gap-6 mb-6">
                <div className="p-3 rounded-lg bg-ai-accent/10 text-ai-accent glow-ai">
                  {step.icon}
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold mb-3 text-foreground">{step.title}</h3>
                  <p className="text-muted-foreground text-lg leading-relaxed">{step.description}</p>
                </div>
              </div>
              
              {step.code && (
                <div className="mb-6">
                  <CodeBlock 
                    code={step.code}
                    language="python"
                    title="Implementation"
                  />
                </div>
              )}
              
              {step.tips && (
                <div className="bg-muted/30 rounded-lg p-6 border border-ai-accent/20">
                  <h4 className="font-semibold text-ai-accent mb-3 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Pro Tips
                  </h4>
                  <ul className="space-y-2">
                    {step.tips.map((tip, tipIndex) => (
                      <li key={tipIndex} className="text-sm text-muted-foreground flex items-start gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-ai-primary mt-2 flex-shrink-0" />
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};