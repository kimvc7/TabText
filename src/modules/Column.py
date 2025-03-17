class Column(object):
    """
    Column module containing functionality to convert feature values into written text
    
    name: String corresponding to the name of the column in the data set.
    attribute: String corresponging to the text description of the column.
    col_type: The type of the column: binary, categorical or numerical.
    verb: The verb required to conjugate the attribute.
    encode_fn: The function used to encode categorical values of this column.
    
    """
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
    
    def create_sentence(self, value, imp_value, prefix, missing_word, replace_numbers, descriptive):
        """
        Parameters:: 
            value: The value of this column at a specific data point
            imp_value: The imputed value of this column at a specific data point.
            prefix: String containing the desired prefix to add at the beginning of the sentence ("", "the Patient", etc.)
            missing_word: String describing how to handle missing values (e.g. "", "is missing" "imp_replace") 
            replace_numbers: Boolean indicating weather or not to replace numerical values with text (e.g. very low, high, normal)
            descriptive: Boolean indicating weather or not the sentence should be descriptive.
        
        Returns::
            String with a sentence describing the column and its value.
            In the case of missing values:
                1. If missing_word == "" the sentence is the empty string
                2. If missing_word == "imp_replace" the sentence is constructed using the imputed value
                3. For all other cases the sentence is constructed using the text in the string missing_word
                
        """
        if descriptive:
            return self.create_descriptive_sentence(value, imp_value, prefix, missing_word, replace_numbers)
        else:
            return self.create_basic_sentence(value, imp_value, prefix, missing_word, replace_numbers)


class Binary_Column(Column):
    """
    Binary Column submodule for columns with values in [1, 0, true, false, "1", "0", "true", "false"]
    
    verb: The positive from of the verb used to conjugate the attribute when value is 1, "1" or "True"
    neg_verb: Negative form of the verb used to conjugate the attribute when value is 0, "0" or "false"
    
    """
    def __init__(self, name, attribute, verb, neg_verb, encode_fn=None):
        self.neg_verb = neg_verb
        super().__init__(name, attribute, "binary", verb, encode_fn)
        

    def create_descriptive_sentence(self, value, imp_value, prefix, missing_word, replace_numbers):
        sentence = ""
        if str(value).lower()  in ["1", "0", "true", "false", "0.0", "1.0"]:
            if int(value) == 1:
                sentence = prefix + self.verb + " " + self.attribute
            elif int(value) == 0:
                sentence = prefix + self.neg_verb + " " + self.attribute
        return sentence
            

    def create_basic_sentence(self, value, imp_value, prefix, missing_word, replace_numbers):
        sentence = ""
        if str(value).lower()  in ["1", "0", "true", "false", "0.0", "1.0"]:
            if int(value) == 1:
                sentence = self.verb + " " + self.attribute + ": yes" 
            elif int(value) == 0:
                sentence = self.neg_verb + " " + self.attribute +" : no"
        elif missing_word != "":
            sentence = self.verb + " " + self.attribute + ": " + missing_word
        return sentence

class Categorical_Column(Column):
    """
    Categorical Column submodule for columns with non-numerical values
    
    """
    def __init__(self, name, attribute, verb, encode_fn=None):
        super().__init__(name, attribute, "categorical", verb, encode_fn)

    def create_descriptive_sentence(self, value, imp_value, prefix, missing_word, replace_numbers):
        if len(prefix) != 0:
            prefix = prefix[:-1] + "'s "
        sentence = ""
        if str(value).lower() not in ["nan", "", "none", "missing"]:
            sentence = prefix + self.attribute + " " + self.verb + " " + str(value)
        elif missing_word not in ["", "imp_replace"]:
            sentence = prefix + self.attribute + " " + self.verb + " " + missing_word
        elif missing_word == "imp_replace":
            sentence = prefix + self.attribute + " " + self.verb + " " + str(imp_value)
        return sentence
            

    def create_basic_sentence(self, value, imp_value, prefix, missing_word, replace_numbers):
        sentence = ""
        if  str(value).lower() not in ["nan", "", "none", "missing"]:
            sentence = self.attribute + ": " + str(value)
        elif missing_word not in ["", "imp_replace"]:
            sentence = self.attribute + ": " + missing_word
        elif missing_word == "imp_replace":
            sentence = self.attribute + ": " + str(imp_value)
        return sentence

class Numerical_Column(Column):
    """
    Numerical Column submodule for columns with numerical values
    
    avg: The average of the values observed for this column (to be computed usign Training set)
    sd: The standard deviation of the values observed for this column (to be computed usign Training set)
    
    """
    def __init__(self, name, attribute, verb, avg, sd, encode_fn = None):
        self.avg = avg
        self.sd = sd
        super().__init__(name, attribute, "numerical", verb, encode_fn)
        
        
    def create_descriptive_sentence(self, value, imp_value, prefix, missing_word, replace_numbers):
        if len(prefix) != 0:
            prefix = prefix[:-1] + "'s "
        sentence = ""
        if str(value).lower() not in ["nan", "", "none", "missing"]:
            value = float(value)
            col_value = self.encode_number(value, replace_numbers)
            sentence = prefix + self.attribute + " " + self.verb + " " + str(col_value) 
        elif  missing_word not in ["", "imp_replace"]:
            sentence = prefix + self.attribute + " " + self.verb + " " + missing_word 
        elif missing_word == "imp_replace":
            col_value = self.encode_number(imp_value, replace_numbers)
            sentence = prefix + self.attribute + " " + self.verb + " " + str(col_value) 
        return sentence
            

    def create_basic_sentence(self, value, imp_value, prefix, missing_word, replace_numbers):
        sentence = ""
        if  str(value).lower() not in ["nan", "", "none", "missing"]:
            value = float(value)
            col_value = self.encode_number(value, replace_numbers)
            sentence = self.attribute + ": " + str(col_value)
        elif missing_word not in ["", "imp_replace"]:
            sentence = self.attribute + ": " + missing_word
        elif missing_word == "imp_replace":
            col_value = self.encode_number(imp_value, replace_numbers)
            sentence = self.attribute + ": " + str(col_value)
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
