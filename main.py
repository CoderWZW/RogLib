from attack import Attack
from data import Data
from utils import read_configuration

# Robustness on Graph data
if __name__ == '__main__':
	# data process

	model_set = {
			"a" : "nettack" 
		}

	model = model_set["a"]
	model_config = read_configuration("./config/data_config/" + model + ".conf")

	

	data = Data(model_config)
	data.load()
	data.process()
	data.train_val_test_split()
	#print(data.split_unlabeled)

	attack_config = read_configuration("./config/attack_config/" + model + ".conf")
	attack = Attack(attack_config, data)
	attack.train_surrogate_model()
	attack.model()
	attack.generate()
	attack.train_base_model_without_perturbations()
	attack.train_base_model_with_perturbations()
	attack.show()
