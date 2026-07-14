def estimate_severity(probabilities: dict) -> dict:
    """Estimate severity based on disease probability scores.
    
    Thresholds:
    - < 0.3: Normal
    - 0.3 - 0.6: Mild
    - 0.6 - 0.8: Moderate
    - >= 0.8: Severe
    """
    max_disease_prob = max(
        probabilities.get('Pneumonia', 0),
        probabilities.get('Tuberculosis', 0)
    )
    
    if max_disease_prob < 0.3:
        level = 'Normal'
        color = '#10B981'
        description = 'No significant findings detected.'
    elif max_disease_prob < 0.6:
        level = 'Mild'
        color = '#F59E0B'
        description = 'Mild abnormality detected. Clinical correlation recommended.'
    elif max_disease_prob < 0.8:
        level = 'Moderate'
        color = '#F97316'
        description = 'Moderate abnormality detected. Further investigation recommended.'
    else:
        level = 'Severe'
        color = '#EF4444'
        description = 'Severe abnormality detected. Immediate clinical attention recommended.'
    
    return {
        'level': level,
        'score': round(max_disease_prob, 4),
        'color': color,
        'description': description
    }
