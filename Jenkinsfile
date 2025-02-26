pipeline {
    agent any
    stages {
        // Stage 1: Set up the virtual environment
        stage('Setup') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                export PIP_BREAK_SYSTEM_PACKAGES=1
                pip install --upgrade pip
                pip install -r ./requirements.txt

                echo "Environment configured successfully!"
                '''
            }
        }

        // Stage 2: Prepare the data
        stage('Prepare Data') {
            steps {
                script {
                    echo '📊 Preparing data...'
                    sh '''
                        . venv/bin/activate
                        python3 model_pipeline.py --prepare
                    '''
                    echo '✅ Data prepared successfully!'
                }
            }
        }

        // Stage 3: Train the model
        stage('Train Model') {
            steps {
                script {
                    echo '🚀 Training the model...'
                    sh '''
                        . venv/bin/activate
                        python3 model_pipeline.py --train
                    '''
                    echo '✅ Model trained successfully!'
                }
            }
        }

        // Stage 4: Evaluate the model
        stage('Evaluate Model') {
            steps {
                script {
                    echo '📊 Evaluating the model...'
                    sh '''
                        . venv/bin/activate
                        python3 model_pipeline.py --evaluate
                    '''
                    echo '✅ Model evaluated successfully!'
                }
            }
        }

        // Stage 5: Save the trained model
        stage('Save Model') {
            steps {
                script {
                    echo '💾 Saving the trained model...'
                    sh '''
                        . venv/bin/activate
                        python3 model_pipeline.py --save
                    '''
                    echo '✅ Model saved successfully!'
                }
            }
        }
    }

    post {
        always {
            echo '🧹 Cleaning up...'
            sh 'rm -rf venv'
            echo '✅ Cleanup completed!'
        }
    }
}