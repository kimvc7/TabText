class Column(object):
    def __init__(self, name, attribute=None, col_type=None, verb=None, encode_fn=None):
        self.name = name
        self.attribute = attribute
        self.type = col_type
        self.verb = verb
        self.encode_fn = encode_fn

    def is_binary(self):
        return self.type == "binary"
    
    def is_categorical(self):
        return self.type == "categorical"
    
    def is_numerical(self):
        return self.type == "numerical"
    
    def create_sentence(self, value, prefix, missing_word, replace_numbers, descriptive):
        if descriptive:
            return self.create_descriptive_sentence(value, prefix, missing_word, replace_numbers)
        else:
            return self.create_basic_sentence(value, prefix, missing_word, replace_numbers)
        
        
class Binary_Column(Column):
    def __init__(self, name, attribute, verb, neg_verb, encode_fn=None):
        self.neg_verb = neg_verb
        super().__init__(name, attribute, "binary", verb, encode_fn)
        

    def create_descriptive_sentence(self, value, prefix, missing_word, replace_numbers):
        sentence = ""
        if str(value).lower()  in ["1", "0", "true", "false"]:
            if int(value) == 1:
                sentence = prefix + self.verb + " " + self.attribute
            elif int(value) == 0:
                sentence = prefix + self.neg_verb + " " + self.attribute
        return sentence
            

    def create_basic_sentence(self, value, prefix, missing_word, replace_numbers):
        sentence = ""
        if str(value).lower()  in ["1", "0", "true", "false"]:
            if int(value) == 1:
                sentence = self.verb + " " + self.attribute + ": yes" 
            elif int(value) == 0:
                sentence = self.neg_verb + " " + self.attribute +" : no"
        elif missing_word != "":
            sentence = self.verb + " " + self.attribute + ": " + missing_word
        return sentence
        
class Categorical_Column(Column):
    def __init__(self, name, attribute, verb, encode_fn=None):
        super().__init__(name, attribute, "categorical", verb, encode_fn)

    def create_descriptive_sentence(self, value, prefix, missing_word, replace_numbers):
        if len(prefix) != 0:
            prefix = prefix[:-1] + "'s "
        sentence = ""
        if str(value).lower() not in ["nan", "", "none", "missing"]:
            sentence = prefix + self.attribute + " " + self.verb + " " + str(value)
        elif  missing_word != "":
            sentence = prefix + self.attribute + " " + self.verb + " " + missing_word
        return sentence
            

    def create_basic_sentence(self, value, prefix, missing_word, replace_numbers):
        if len(prefix) != 0:
            prefix = prefix[:-1] + "'s "
        sentence = ""
        if  str(value).lower() not in ["nan", "", "none", "missing"]:
            sentence = self.attribute + ": " + str(value)
        elif missing_word != "":
            sentence = self.attribute + ": " + missing_word
        return sentence
    
class Numerical_Column(Column):
    def __init__(self, name, attribute, verb, avg, sd, encode_fn = None):
        self.avg = avg
        self.sd = sd
        super().__init__(name, attribute, "numerical", verb, encode_fn)
        
        
    def create_descriptive_sentence(self, value, prefix, missing_word, replace_numbers):
        if len(prefix) != 0:
            prefix = prefix[:-1] + "'s "
        sentence = ""
        if str(value).lower() not in ["nan", "", "none", "missing"]:
            col_value = self.encode_number(value, replace_numbers)
            sentence = prefix + self.attribute + " " + self.verb + " " + str(col_value) 
        elif  missing_word != "":
            sentence = prefix + self.attribute + " " + self.verb + " " + missing_word 
        return sentence
            

    def create_basic_sentence(self, value, prefix, missing_word, replace_numbers):
        if len(prefix) != 0:
            prefix = prefix[:-1] + "'s "
        sentence = ""
        if  str(value).lower() not in ["nan", "", "none", "missing"]:
            col_value = self.encode_number(value, replace_numbers)
            sentence = self.attribute + ": " + str(col_value)
        elif missing_word != "":
            sentence = self.attribute + ": " + missing_word
        return sentence
    
    def encode_number(self, value, replace_numbers):
        new_value = value
        if replace_numbers:
            if self.avg - 2*self.sd > value:
                new_value = "very low"
            elif self.avg - 2*self.sd <= value < self.avg - self.sd:
                new_value = "low"
            elif self.avg + 2*self.sd >= value > self.avg + self.sd:
                new_value = "high"
            elif self.avg + 2*self.sd < value:
                new_value = "very high"
            else:
                new_value = "normal"
        return new_value
