"""
Functions to read and use a database of amino acid properties.
@author Michael Hamilton, Colorado State University, 2007

database citations:

*Nakai, K., Kidera, A., and Kanehisa, M.; Cluster analysis of amino acid indices for
prediction of protein structure and function. Protein Eng. 2, 93-100 (1988). [PMID:3244698]

*Tomii, K. and Kanehisa, M.; Analysis of amino acid indices and mutation matrices for
sequence comparison and structure prediction of proteins. Protein Eng. 9, 27-36 (1996). [PMID:9053899]

*Kawashima, S., Ogata, H., and Kanehisa, M.; AAindex: amino acid index database.
Nucleic Acids Res. 27, 368-369 (1999). [PMID:9847231]

*Kawashima, S. and Kanehisa, M.; AAindex: amino acid index database.
Nucleic Acids Res. 28, 374 (2000). [PMID:10592278]
"""

import numpy, os
from PyML import SparseDataSet
from PyML.preproc import preproc

class AA_Property:
    """
    Class to maintain information and access to an AA property.
    """
    def __init__( self, d, aa_dict ):
        """
        Arguments:
        `header` - a list containing the following values:
        `H` - header id
        `D` - description
        `R` - reference
        `A` - author
        `T` - title
        `J` - publisher
        `C` - cross links to other AA properties with high correlation, a dictionary of H->correlation pairs
        """
        self.H = d[ 'H' ]
        self.D = d[ 'D' ]
        self.R = d[ 'R' ]
        self.A = d[ 'A' ]
        self.T = d[ 'T' ]
        self.J = d[ 'J' ]
        self.C = d[ 'C' ]
        self.add_values( aa_dict )

    def __repr__( self ):
        return self.H

    def add_values( self, d ):
        """
        Add individual aa property values
        """
        self.prop_map = d
        self.prop_avg = {}
        for key in d:
            self.prop_avg = numpy.mean( self.prop_map.values () )
            
    def calc_avg( self, seq ):
        curr_sum = 0.0
        keys = self.prop_map.keys()
        for aa in seq:
            aa = aa.upper()
            if aa not in keys:
                #print "Yikes", aa
                curr_sum += self.prop_avg
            else:
                curr_sum += self.prop_map[ aa ]
        return curr_sum / len( seq )

        
    
def load_db( fname = 'aa_property.db', remove_correlated=True ):
    db = open( fname, 'r' )
    properties = _get_properties_from_file( db )
    if remove_correlated:
        d = {}
        for prop in properties: d[ prop.H ] = prop
        for prop in properties:
            for cor in prop.C:
                d.pop( cor, None )                    
        properties = []        
        for prop in d:
            properties.append( d[ prop ] )
    return properties

def _get_current_info( db, sentinel, cval ):

    val = cval
    
    while( True ):
        line = db.next().strip()
        if line.startswith( sentinel ):
            return val[ 1: ].strip(), line
        val += " " +  line 

def _get_cross_refs( db, cval ):
    val = cval
    sentinel = "I "
    line = ''
    while( True ):
        line = db.next().strip()
        if line.startswith( sentinel ):
            break
        val += " " + line

    val = val.strip()
    raw_vals = val[1:].split()
    val_dict = {}
    
    for i in xrange( 0, len( raw_vals ), 2 ):
        val_dict[ str( raw_vals[ i ] ) ] = raw_vals[ i+1 ]

    return val_dict, line

def _get_curr_props( db ):
    val = ''
    keys = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    sentinel = "//"
    while( True ):
        line = db.next()
        if line.startswith( sentinel ):
            break
        val += " " + line
    val = val.strip()
    val = val.split()
    val = [ float( token ) for token in val ]
    return dict( zip( keys, val ))
    
def _get_properties_from_file( db ):
    properties = []
    cval = db.next().strip()
    while (True):
        h, cval = _get_current_info( db, 'D ', cval )
        d, cval = _get_current_info( db, 'R ', cval )
        r, cval = _get_current_info( db, 'A ', cval )
        a, cval = _get_current_info( db, 'T ', cval )
        t, cval = _get_current_info( db, 'J ', cval )
        j, cval = _get_current_info( db, 'C ', cval )
        c, cval = _get_cross_refs( db, cval )
        aa_dict = _get_curr_props( db )
        hdict = { 'H' : h, 'D' : d, 'R' : r, 'A' : a, 'T' : t, 'J' : j, 'C' : c }
        properties.append( AA_Property( hdict, aa_dict ) )
        try:
            cval = db.next().strip()
        except StopIteration:
            break
            
        
    
    return properties    

class aa_property_generator (object) :

    def __init__(self, select_features = False) :
        self.database = load_db('aa_property.db', select_features )

    def get_property_avgs(self, sequence) :
        values = [property.calc_avg(sequence) for property in self.database]

    def get_property_avgs_as_dict(self, sequence) :
        values = {}
        for property in self.database :
            values[property.D] = property.calc_avg(sequence)
        return values
    
    def get_property_descriptions(self) :

        return [property.D for property in self.database]

    def property_data(self, sequences) :

        prop_generator = aa_property_generator()
        data_matrix = [prop_generator.get_property_avgs_as_dict(sequence) for sequence in sequences]
        data = SparseDataSet(data_matrix)
        standardizer = preproc.Standardizer()
        standardizer.train(data)

        return data

def test() :
    props = load_db('aa_property.db', False )
    peptide = "MAVYCYALNSLVIMNSANEMKSGGGPGPSGSETPPPPRRAVLSPGSVFSPGRG"
    property_avgs = [ ]
    property_names = []
    for prop in props:
        curr_avg = prop.calc_avg( peptide )
        property_names.append(prop.D)
        property_avgs.append( curr_avg )
    property_avgs = numpy.array( property_avgs )
    return property_avgs


if __name__ == "__main__":
    test()
