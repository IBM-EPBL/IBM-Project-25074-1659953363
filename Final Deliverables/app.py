# Import Libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)#initiate flask app

def load_model(file='model.sav'):#load the saved model
	return pickle.load(open(file, 'rb'))

@app.route('/')
def index():#main page
	return render_template('car.html')

@app.route('/predict_page')
def predict_page():#predicting page
	return render_template('value.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
	reg_year = int(request.args.get('regyear'))
	powerps = float(request.args.get('powerps'))
	kms= float(request.args.get('kms'))
	reg_month = int(request.args.get('regmonth'))

	gearbox = request.args.get('geartype')
	damage = request.args.get('damage')
	model = request.args.get('model')
	brand = request.args.get('brand')
	fuel_type = request.args.get('fuelType')
	veh_type = request.args.get('vehicletype')

	new_row = {'yearOfReg':reg_year, 'powerPS':powerps, 'kilometer':kms,
				'monthOfRegistration':reg_month, 'gearbox':gearbox,
				'notRepairedDamage':damage,
				'model':model, 'brand':brand, 'fuelType':fuel_type,
				'vehicletype':veh_type}

	labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicletype']
	keys=[['manual','automatic','not-declared'],['yes','not-declared','no'],['not-declared','grand','golf','fabia','3er','2_reihe','c_max','3_reihe','passat','navara','polo','twingo','a_klasse','scirocco','5er','andere','civic','punto','e_klasse','clio','kadett','one','fortwo','1er','b_klasse','a8','jetta','c_klasse','micra','vito','sprinter','astra','156','escort','forester','xc_reihe','fiesta','scenic','ka','a1','transporter','focus','a4','tt','a6','jazz','omega','slk','7er','combo','corsa','80','147','glk','z_reihe','sorento','ibiza','mustang','eos','touran','getz','insignia','almera','megane','a3','r19','caddy','mondeo','cordoba','colt','impreza','vectra','lupo','berlingo','m_klasse','tiguan','6_reihe','c4','panda','up','i_reihe','ceed','kangoo','5_reihe','yeti','octavia','zafira','mii','rx_reihe','6er','modus','fox','matiz','beetle','rio','touareg','logan','spider','cuore','s_max','a2','x_reihe','a5','galaxy','c3','viano','s_klasse','1_reihe','sharan','avensis','sl','roomster','q5','santa','leon','cooper','4_reihe','sportage','laguna','ptcruiser','clk','primera','espace','exeo','159','transit','juke','v40','carisma','accord','corolla','lanos','phaeton','boxster','verso','rav','kuga','qashqai','swift','picanto','superb','stilo','alhambra','911','m_reihe','roadster','ypsilon','galant','justy','90','sirion','signum','crossfire','agila','duster','v50','mx_reihe','meriva','discovery','c_reihe','v_klasse','yaris','c5','aygo','seicento','cc','carnival','fusion','bora','cl','tigra','300c','500','100','q3','cr_reihe','spark','x_type','ducato','s_type','x_trail','toledo','altea','voyager','calibra','v70','bravo','range_rover','forfour','tucson','q7','c1','citigo','jimny','cx_reihe','cayenne','wrangler','lybra','range_rover_sport','lancer','freelander','captiva','range_rover_evoque','sandero','note','antara','900','defender','cherokee','clubman','arosa','legacy','pajero','auris','c2','niva','s60','nubira','vivaro','g_klasse','lodgy','850','serie_2','charade','croma','outlander','gl','kaefer','doblo','musa','amarok','9000','kalos','v60','200','145','b_max','delta','aveo','rangerover','move','materia','terios','kalina','elefantino','i3','samara','kappa','serie_3','discovery_sport'],['audi','jeep','volkswagen','skoda','bmw','peugeot','ford','mazda','nissan','renault','mercedes_benz','honda','fiat','opel','mini','smart','hyundai','alfa_romeo','subaru','volvo','mitsubishi','kia','seat','lancia','porsche','citroen','toyota','chevrolet','dacia','suzuki','daihatsu','chrysler','sonstige_autos','jaguar','daewoo','rover','saab','land_rover','lada','trabant'],['diesel','petrol','not-declared','lpg','others','hybrid','cng','electric'],['coupe','suv','small car','limousine','convertible','bus','combination','not-declared','others']]
	values = [[1,0,2],[1,2,0],[162,118,117,102,11,8,60,10,171,160,174,227,32,200,15,39,73,177,96,75,130,167,107,6,47,31,125,59,152,237,213,42,3,99,105,244,103,199,129,25,224,104,28,225,30,124,166,208,18,79,83,19,2,116,248,209,121,158,98,222,114,123,36,150,27,182,62,155,81,78,122,234,144,50,146,219,16,57,170,228,120,69,134,14,246,165,249,153,191,17,154,108,149,49,188,221,143,211,87,194,26,241,29,113,56,236,193,5,204,44,207,190,179,198,141,80,12,212,137,176,76,175,100,101,4,223,127,229,65,33,82,139,172,52,235,187,136,181,216,173,215,214,35,24,147,189,247,112,128,21,206,205,86,34,95,230,159,151,91,61,233,245,58,46,201,68,66,110,51,74,218,9,13,0,178,84,210,243,94,195,242,220,37,239,63,232,53,183,106,226,180,54,72,126,88,67,240,145,185,138,109,64,184,197,163,40,22,89,71,77,41,140,169,43,55,161,192,164,238,111,142,20,202,70,85,168,115,131,93,157,38,23,133,231,7,1,48,90,45,186,156,148,217,132,97,119,196,135,203,92],[1,14,38,31,2,25,10,19,23,27,20,11,9,24,21,32,12,0,34,39,22,15,30,17,26,5,36,3,6,35,8,4,33,13,7,28,29,18,16,37],[1,7,5,4,6,3,0,2],[3,8,7,4,2,0,1,5,6]]
	hashDict=[]
	for i in range(len(labels)):
		hashDict.append(dict(zip(keys[i], values[i])))
	for i in range(len(labels)):
		new_row[labels[i]]=hashDict[i][new_row[labels[i]]]
	new_df = pd.DataFrame(columns=['vehicletype','yearOfReg','gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType','brand','notRepairedDamage'])
	new_df = new_df.append(new_row, ignore_index=True)

	X = new_df.values.tolist()
	print('\n\n', X ,'\n\n')
	predict = reg_model.predict(X)

	#predict = predictions['predictions'][0]['values'][0][0]
	print("Final prediction :",predict)

	return render_template('predict.html',predict=predict)

if __name__=='__main__':
	reg_model = load_model()#load the saved model
	app.run(debug=True)