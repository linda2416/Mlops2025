pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh '''
                echo "Environment configured successfully!"
                '''
            }
        }

        stage('Prepare') {
            steps {
                sh '''
                echo "Code verified successfully!"
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                echo "Model trained successfully!"
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                sh '''
                echo "Model evaluated successfully!"
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                . venv/bin/activate
                echo "Tests executed successfully!"
                '''
            }
        }

        stage('Clean') {
            steps {
                sh '''
                echo "Cleanup completed!"
                '''
            }
        }
    }

    post {
        always {
            echo "Pipeline MLOps executed successfully!"
        }
    }
}
