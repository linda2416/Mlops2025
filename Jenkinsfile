pipeline {
    agent any

    triggers {
        githubPush()
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo '🚀 Récupération du code source...'
                git branch: 'main', url: 'https://github.com/votre-utilisateur/mlops-pipeline.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo '🐳 Construction de l\'image Docker...'
                sh 'docker build -t lindadocker24/linda_trabelsi_4ds5_mlops .'
            }
        }

        stage('Push Docker Image') {
            steps {
                echo '⏫ Push de l\'image sur Docker Hub...'
                withCredentials([usernamePassword(credentialsId: "docker-hub-credentials", usernameVariable: "DOCKER_USER", passwordVariable: "DOCKER_PASS")]) {
                    sh '''
                    echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                    docker push lindadocker24/linda_trabelsi_4ds5_mlops
                    docker logout
                    '''
                }
            }
        }

        stage('Deploy Container') {
            steps {
                echo '🚀 Déploiement du conteneur...'
                sh '''
                docker stop mlops_container || true
                docker rm mlops_container || true
                docker run -d -p 8000:8000 --name mlops_container lindadocker24/linda_trabelsi_4ds5_mlops
                '''
            }
        }
    }

    post {
        always {
            echo '🧹 Nettoyage...'
            sh 'docker system prune -f'
        }
    }
}

