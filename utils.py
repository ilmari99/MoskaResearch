CARD_VALUES = tuple(range(2,15))                            # Initialize the standard deck
CARD_SUITS = ("C","D","H","S") 
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣'}    #Conversion table
MAIN_DECK = None                                            # The main deck

def suit_to_symbol(suits):
    if isinstance(suits,str):
        return CARD_SUIT_SYMBOLS[suits]
    return [CARD_SUIT_SYMBOLS[s] for s in suits]

def check_can_fall_card(played_card, fall_card,triumph):
    """ Returns true, if the played_card, can fall the fall_card"""
    success = False
    # Jos kortit ovat samaa maata ja pelattu kortti on suurempi
    if played_card.suit == fall_card.suit and played_card.value > fall_card.value:
        success = True
    # Jos pelattu kortti on valttia, ja kaadettava kortti ei ole valttia
    elif played_card.suit == triumph and fall_card.suit != triumph:
            success = True
    return success

class TurnCycle:
    population = []
    ptr = 0
    def __init__(self,population,ptr=0):
        self.population = population
        self.ptr = ptr
        
    def get_at_index(self,index = None):
        if index is None:
            index = self.ptr
        #print(f"Returning {index % len(self.population)}")
        return self.population[index % len(self.population)]
        
    def get_next(self, incr_ptr = True):
        out = self.get_at_index(self.ptr + 1)
        if incr_ptr:
            self.ptr += 1
        return out
    
    def get_prev(self, incr_ptr = True):
        out = self.get_at_index(self.ptr - 1)
        if incr_ptr:
            self.ptr -= 1
        return out
    
    def get_prev_condition(self, cond, incr_ptr=False):
        """ Returns the previous element in the cycle, that matches the condition"""
        count = 1
        og_count = int(self.ptr)
        nxt = self.get_prev()
        while not cond(nxt):
            nxt = self.get_prev()
            if count == len(self.population):
                nxt = []
                break
            count += 1
        if not incr_ptr:
            self.set_pointer(og_count)
        return nxt
    
    def add_to_population(self,val,ptr=None):
        self.population.append(val)
        self.ptr += 1
        if ptr:
            self.ptr = ptr
    
    def get_next_condition(self,cond = lambda x : True, incr_ptr=True):
        """ Returns the next element in the cycle, that matches the condition"""
        count = 1
        og_count = int(self.ptr)
        nxt = self.get_next()
        while not cond(nxt):
            nxt = self.get_next()
            if count == len(self.population):
                nxt = []
                break
            count += 1
        if not incr_ptr:
            self.set_pointer(og_count)
        return nxt
    
    def set_pointer(self,ptr):
        self.ptr = ptr
        
        
if __name__ == "__main__":
    tc = TurnCycle([0,1,2,3,4,5])
    print(tc.get_at_index())
    print(tc.get_next_condition(lambda x : x ==4))
    print(tc.ptr)
    print(tc.get_prev_condition(lambda x : x == 0))
    print(tc.ptr)
