from django.shortcuts import render
from django.http import JsonResponse
import pickle
import numpy as np

# Charger le modèle
MODEL_PATH = "/home/linda_wsl/Linda_Farah_Trabelsi_4ds5_ml_project/model.pkl"

# Charger le modèle avec pickle
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

def predict_view(request):
    prediction = None
    if request.method == "POST":
        try:
            # Récupérer les données du formulaire directement depuis request.POST
            account_length = float(request.POST['account_length'])
            international_plan = int(request.POST['international_plan'])
            voice_mail_plan = int(request.POST['voice_mail_plan'])
            number_vmail_messages = float(request.POST['number_vmail_messages'])
            total_day_minutes = float(request.POST['total_day_minutes'])
            total_day_calls = float(request.POST['total_day_calls'])
            total_day_charge = float(request.POST['total_day_charge'])
            total_eve_minutes = float(request.POST['total_eve_minutes'])
            total_eve_calls = float(request.POST['total_eve_calls'])
            total_night_minutes = float(request.POST['total_night_minutes'])
            total_night_calls = float(request.POST['total_night_calls'])
            total_intl_minutes = float(request.POST['total_intl_minutes'])
            total_intl_calls = float(request.POST['total_intl_calls'])
            customer_service_calls = float(request.POST['customer_service_calls'])

            # Convertir les données en tableau numpy pour la prédiction
            input_data = np.array([
                account_length,
                international_plan,
                voice_mail_plan,
                number_vmail_messages,
                total_day_minutes,
                total_day_calls,
                total_day_charge,
                total_eve_minutes,
                total_eve_calls,
                total_night_minutes,
                total_night_calls,
                total_intl_minutes,
                total_intl_calls,
                customer_service_calls
            ]).reshape(1, -1)

            # Faire la prédiction
            prediction = model.predict(input_data)[0]
        except KeyError as e:
            return JsonResponse({"error": f"Missing data for: {str(e)}"}, status=400)

    return render(request, 'predict/predict_form.html', {'prediction': prediction})
