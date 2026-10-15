"""
VitaFusion AI — Hospital database indexed by city (lowercase).
"""

HOSPITALS: dict[str, list[dict]] = {
    "chennai": [
        {"id": "ch1", "name": "Apollo Hospitals, Greams Road", "address": "21 Greams Lane, Chennai 600006", "phone": "044-28290200", "specialty": "Multi-specialty", "rating": 4.8},
        {"id": "ch2", "name": "Fortis Malar Hospital", "address": "52 1st Main Rd, Gandhi Nagar, Chennai 600020", "phone": "044-42892222", "specialty": "Cardiac Care", "rating": 4.7},
        {"id": "ch3", "name": "MIOT International", "address": "4/112 Mount Poonamallee Rd, Chennai 600089", "phone": "044-42002288", "specialty": "Orthopaedics & Cardiology", "rating": 4.6},
        {"id": "ch4", "name": "Sri Ramachandra Medical Centre", "address": "No.1, Ramachandra Nagar, Porur, Chennai 600116", "phone": "044-45928888", "specialty": "Multi-specialty", "rating": 4.7},
        {"id": "ch5", "name": "Kauvery Hospital", "address": "199, Luz Church Road, Mylapore, Chennai 600004", "phone": "044-40001000", "specialty": "Emergency & Critical Care", "rating": 4.5},
    ],
    "mumbai": [
        {"id": "mu1", "name": "Lilavati Hospital", "address": "A-791 Bandra Reclamation, Bandra West, Mumbai 400050", "phone": "022-26751000", "specialty": "Multi-specialty", "rating": 4.8},
        {"id": "mu2", "name": "Kokilaben Dhirubhai Ambani Hospital", "address": "Rao Saheb Achutrao Patwardhan Marg, Mumbai 400053", "phone": "022-42696969", "specialty": "Cardiac & Oncology", "rating": 4.9},
        {"id": "mu3", "name": "Hinduja Hospital", "address": "Veer Savarkar Marg, Mahim, Mumbai 400016", "phone": "022-24452222", "specialty": "Multi-specialty", "rating": 4.7},
        {"id": "mu4", "name": "Nanavati Max Super Speciality", "address": "S.V. Road, Vile Parle West, Mumbai 400056", "phone": "022-26100000", "specialty": "Cardiac & Neuro", "rating": 4.6},
    ],
    "delhi": [
        {"id": "dl1", "name": "AIIMS New Delhi", "address": "Sri Aurobindo Marg, Ansari Nagar, New Delhi 110029", "phone": "011-26588500", "specialty": "Multi-specialty", "rating": 4.9},
        {"id": "dl2", "name": "Apollo Hospitals, Sarita Vihar", "address": "Sarita Vihar, Mathura Road, New Delhi 110076", "phone": "011-71791090", "specialty": "Multi-specialty", "rating": 4.8},
        {"id": "dl3", "name": "Max Super Speciality Hospital, Saket", "address": "1 Press Enclave Road, Saket, New Delhi 110017", "phone": "011-26515050", "specialty": "Cardiac & Neurology", "rating": 4.7},
        {"id": "dl4", "name": "Fortis Escorts Heart Institute", "address": "Okhla Road, New Delhi 110025", "phone": "011-47135000", "specialty": "Cardiac", "rating": 4.8},
    ],
    "bangalore": [
        {"id": "bl1", "name": "Manipal Hospital, Old Airport Road", "address": "98 HAL Airport Road, Bangalore 560017", "phone": "080-25024444", "specialty": "Multi-specialty", "rating": 4.8},
        {"id": "bl2", "name": "Narayana Health City", "address": "258/A Bommasandra Industrial Area, Bangalore 560099", "phone": "080-71222222", "specialty": "Cardiac Care", "rating": 4.7},
        {"id": "bl3", "name": "Fortis Hospital Bannerghatta", "address": "154/9 Bannerghatta Road, Bangalore 560076", "phone": "080-66214444", "specialty": "Multi-specialty", "rating": 4.6},
        {"id": "bl4", "name": "Sakra World Hospital", "address": "Devarabeesanahalli, Varthur Hobli, Bangalore 560103", "phone": "080-49690000", "specialty": "Emergency & Critical Care", "rating": 4.7},
    ],
    "hyderabad": [
        {"id": "hy1", "name": "Apollo Hospitals, Jubilee Hills", "address": "Jubilee Hills, Hyderabad 500033", "phone": "040-23607777", "specialty": "Multi-specialty", "rating": 4.8},
        {"id": "hy2", "name": "KIMS Hospital", "address": "1-8-31/1, Minister Road, Secunderabad 500003", "phone": "040-44885000", "specialty": "Multi-specialty", "rating": 4.7},
        {"id": "hy3", "name": "Yashoda Hospitals", "address": "Alexander Road, Secunderabad 500003", "phone": "040-45671234", "specialty": "Cardiac & Neurology", "rating": 4.6},
        {"id": "hy4", "name": "Care Hospitals", "address": "Road No.1, Banjara Hills, Hyderabad 500034", "phone": "040-30410000", "specialty": "Emergency & Trauma", "rating": 4.5},
    ],
    "kolkata": [
        {"id": "ko1", "name": "Apollo Gleneagles Hospital", "address": "58 Canal Circular Road, Kolkata 700054", "phone": "033-23203040", "specialty": "Multi-specialty", "rating": 4.7},
        {"id": "ko2", "name": "Fortis Hospital Anandapur", "address": "730 Anandapur, E.M. Bypass, Kolkata 700107", "phone": "033-66284444", "specialty": "Multi-specialty", "rating": 4.6},
        {"id": "ko3", "name": "Medica Superspecialty Hospital", "address": "127 Mukundapur, E.M. Bypass, Kolkata 700099", "phone": "033-65000600", "specialty": "Cardiac & Neuro", "rating": 4.5},
    ],
    "pune": [
        {"id": "pu1", "name": "Jehangir Hospital", "address": "32 Sassoon Road, Pune 411001", "phone": "020-66810000", "specialty": "Multi-specialty", "rating": 4.7},
        {"id": "pu2", "name": "Ruby Hall Clinic", "address": "40 Sassoon Road, Pune 411001", "phone": "020-66455555", "specialty": "Cardiac & Ortho", "rating": 4.6},
        {"id": "pu3", "name": "Sahyadri Hospitals", "address": "30-C, Karve Road, Deccan Gymkhana, Pune 411004", "phone": "020-67213000", "specialty": "Multi-specialty", "rating": 4.6},
    ],
    "ahmedabad": [
        {"id": "ah1", "name": "Apollo Hospitals, Ahmedabad", "address": "Plot No.1A, GIDC Estate, Bhat, Gandhinagar 382428", "phone": "079-66701800", "specialty": "Multi-specialty", "rating": 4.7},
        {"id": "ah2", "name": "Sterling Hospital", "address": "Off. Gurukul Road, Memnagar, Ahmedabad 380052", "phone": "079-40012000", "specialty": "Multi-specialty", "rating": 4.6},
    ],
}

_FALLBACK_CITY = "chennai"


def get_hospitals_for_city(city: str) -> list[dict]:
    """Return hospital list for a given city; falls back to Chennai."""
    key = city.lower().strip()
    return HOSPITALS.get(key, HOSPITALS[_FALLBACK_CITY])


def get_hospital_by_id(hospital_id: str) -> dict | None:
    """Lookup a hospital by its ID across all cities."""
    for hospitals in HOSPITALS.values():
        for h in hospitals:
            if h["id"] == hospital_id:
                return h
    return None


SUPPORTED_CITIES = sorted(HOSPITALS.keys())
