/**********************************************************************
MI-1 - Multiple instance learning of calmodulin binding sites by Fayyaz-ul-Amir Afsar Minhas and Asa Ben-Hur, Colorado State University.					
Version: 2.01 (Uses combination of position dependent and independent amino acid compositions)
Fixed a minor bug in normalization
**********************************************************************/
//function classify(sequence)  is the function that performs the classification
var exampleSequences = 
    {'Ure2': 'MMNNNGNQVSNLSNALRQVNIGNRNSNTTTDQSNINFEFSTGVNNNNNNNSSSNNNNVQNNNSGRNGSQNNDNENNIKNTLEQHRQQQQAFSDMSHVEYSRITKFFQEQPLEGYTLFSHRSAPNGFKVAIVLSELGFHYNTIFLDFNLGEHRAPEFVSVNPNARVPALIDHGMDNLSIWESGAILLHLVNKYYKETGNPLLWSDDLADQSQINAWLFFQTSGHAPMIGQALHFRYFHSQKIASAVERYTDEVRRVYGVVEMALAERREALVMELDTENAAAYSAGTTPMSQSRFFDYPVWLVGDKLTIADLAFVPWNNVVDRIGINIKIEFPEVYKWTKHMMRRPAVIKALRGE',
     'Sup35': 'MSDSNQGNNQQNYQQYSQNGNQQQGNNRYQGYQAYNAQAQPAGGYYQNYQGYSGYQQGGYQQYNPDAGYQQQYNPQGGYQQYNPQGGYQQQFNPQGGRGNYKNFNYNNNLQGYQAGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTKPAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEEVDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTIEVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNKMDDPTVNWSKERYDQCVSNVSNFLRAIGYNIKTDVVFMPVSGYSGANLKDHVDPKECPWYTGPTLLEYLDTMNHVDRHINAPFMLPIAAKMKDLGTIVEGKIESGHIKKGQSTLLMPNKTAVEIQNIYNETENEVDMAMCGEQVKLRIKGVEEEDISPGFVLTSPKNPIKSVTKFVAQIAIVELKSIIAAGFSCVMHVHTAIEEVHIVKLLHKLEKGTNRKSKKPPAFAKKGMKVIAVLETEAPVCVETYQDYPQLGRFTLRDQGTTIAIGKIVKIAE',
     'McM1': 'GNDMQRQQPQQQQPQQQQQVLNAHANSLGHLNQDQVPAGALKQEVKSQLLGGANPNQNSMIQQQQHHTQNSQPQQQQQQQPQQQMSQQQMSQHPRPQQGIPHPQQSQPQQQQQQQQQLQQQQQQQQQQPLTGIHQPHQQAFANAASPYLNAEQNAAYQQYFQEPQQGQY',	 'Snf2':'QFAAKQRQELQMQRQQQGISGSQQNIVPNSSDQAELPNNASSHISASASPHLAPNMQLNGNETFSTSAHQSPIMQTQMPLNSNGGNNMLPQRQSSVGSLNATNFSPTPANNGENAAEKPDNSNHNNLNLNNSELQPQNRSLQEHNIQDSNVMPGSQINSPMPQQAQMQQAQFQAQQAQQAQQAQQAQQAQARLQQG'
	 };
//weights (obtained from training )

var WA_1=[[2.194543192994716], [-0.816826318896342], [-4.628877788521357], [-3.225089309350531], [-3.2238671354005937], [0.7598525760535292], [-1.136292831394751], [0.9474436625887771], [-2.30988615881968], [-0.7144523188305107], [-1.529807070064503], [5.5102217163093865], [-0.8021204951838719], [4.339392660312244], [-0.5090873722256528], [0.5546231247989757], [1.648994361216431], [-2.7177499005764787], [1.3844004504560385], [1.3344093175134977]]


var window_size = 41;
var half_window_size = Math.floor(window_size / 2);
var amino_acids = 'ACDEFGHIKLMNPQRSTVWY';
var aaindx={'A' : 0, 'C' : 1, 'D' : 2, 'E' : 3, 'F' : 4,
              'G' : 5, 'H' : 6, 'I' : 7, 'K' : 8, 'L': 9,
              'M' : 10, 'N' : 11, 'P' : 12, 'Q' : 13, 'R' : 14,
              'S' : 15, 'T' : 16, 'V' : 17, 'W' : 18, 'Y' : 19};
			  

function initArray(length, value) {
    var arr = [], i = 0;
    arr.length = length;
    while (i < length) { arr[i++] = value; }
    return arr;
}

function Matrix(ary) {
    this.mtx = ary;
    this.height = ary.length;
    this.width = ary[0].length;
}
Matrix.prototype.setValue = function(r,c,v) {
	if ((this.width <= c) || (this.height <= r) || (c<0) || (r<0)) {
        throw "error: incompatible dimensions";
    }
	this.mtx[r][c]=v;
}

Matrix.prototype.init = function(r,c) {
	//create a rxc matrix with all zeros
    var s = initArray(r,[]);
    for (var i = 0; i < r; i++) 
        s[i]=initArray(c,0);
	this.mtx=s;
	this.height=r;
	this.width=c;
}
Matrix.prototype.initV = function(r,c,v) {
	//create a rxc matrix with all zeros
    var s = initArray(r,[]);
    for (var i = 0; i < r; i++) 
        s[i]=initArray(c,v);
	this.mtx=s;
	this.height=r;
	this.width=c;
}
Matrix.prototype.setRand = function() {
	//set all elements to random values
    for (var i = 0; i < this.width; i++) {        
        for (var j = 0; j < this.height; j++) {
            this.mtx[j][i]=2*Math.random()-1
        }
    }
}

Matrix.prototype.toString = function() {
    var s = []
    for (var i = 0; i < this.mtx.length; i++) 
        s.push( this.mtx[i].join(",") );
    return s.join("\n");
}
Matrix.prototype.sum= function() {
	//return the sum of all elements in the matrix
    var s = 0.0;
    for (var i = 0; i < this.width; i++) {        
        for (var j = 0; j < this.height; j++) {
            s = s+this.mtx[j][i];
        }
    }
    return s;
}
Matrix.prototype.sumColumns = function() {
	//return the sum of all columns for each row
    var sc = [];
    for (var i = 0; i < this.height; i++) {
        sc[i] = [0];
        for (var j = 0; j < this.width; j++) {
            sc[i][0] = sc[i][0]+this.mtx[i][j];
        }
    }
    return new Matrix(sc);
}
Matrix.prototype.ewMult = function(other) {
	//Perform element-wise multiplication
	if ((this.width != other.width) || (this.height != other.height)) {
        throw "error: incompatible sizes for element-wise multiplication";
    } 
    var result = [];
    for (var i = 0; i < this.height; i++) {
        result[i] = [];
        for (var j = 0; j < this.width; j++) {
            result[i][j] = this.mtx[i][j] * other.mtx[i][j];
        }
    }
    return new Matrix(result); 
}
Matrix.prototype.scMult = function(s) {
	//Perform multiplication with a scalar
    var result = [];
    for (var i = 0; i < this.height; i++) {
        result[i] = [];
        for (var j = 0; j < this.width; j++) {
            result[i][j] = this.mtx[i][j] * s;
        }
    }
    return new Matrix(result); 
}
Matrix.prototype.dot = function(other) {
	//element wise-multiplication and then summation	
    return (this.ewMult(other)).sum();
}
Matrix.prototype.norm = function() {
	//return the norm (L2)
    return Math.sqrt(this.dot(this));
}

Matrix.prototype.getNormalized = function() {
	//return the unit norm matrix	
    return this.scMult(1/this.norm());
}
Matrix.prototype.transpose = function() {
    var transposed = [];
    for (var i = 0; i < this.width; i++) {
        transposed[i] = [];
        for (var j = 0; j < this.height; j++) {
            transposed[i][j] = this.mtx[j][i];
        }
    }
    return new Matrix(transposed);
}
function range(a,b) {
    if (typeof(b) == 'undefined') {
        b = a;
        a = 0;        
    }
    var result = [];
    for (var i = a; i < b; i++)
        result.push(i);
    return result;
}
function contains(array,element) {
    return array.indexOf(element) != -1;
}

var WW_1=new Matrix(WA_1)

function window_scores(sequence) {
    //list comprehensions coming soon to JavaScript
    //return [window_score(sequence, i, aa_dict, ignore_consecutive_prolines) for i in range(len(sequence))]
    function oneWindow(i) {
        return window_score(sequence, i);
    }
    return range(sequence.length).map(oneWindow);
}

function window_score(sequence, position) {
    // destructuring assignments coming in JavaScript
    //start,end = get_window(sequence, position)	
    var w = get_window(sequence, position);
    var start = w[0];
    var end = w[1];	
	if(end-start!=window_size)
		return -Infinity
	var FV_pd1=new Matrix([[]])
	FV_pd1.init(20,window_size)
	var cc=0
	for (var i=start; i<end;i++,cc++){
		if (contains(amino_acids,sequence[i]))
			FV_pd1.setValue(aaindx[sequence[i]],cc,1.0);
	}
	var FV_1=FV_pd1.sumColumns()
	FV_1=FV_1.getNormalized()	
	var n=Math.sqrt(FV_1.dot(FV_1))
	FV_1=FV_1.scMult(1.0/n)	
	var score=FV_1.dot(WW_1)-4.30303023051
	return score
}

function get_window(sequence, position) {
    var start = Math.max(position - half_window_size, 0);
    var end = Math.min(sequence.length, position + half_window_size + 1);
    return [start,end];
}                                   

function super_window_scores(sequence, window_scores) {
// Selects max within the window to be the score for that window
    var scores = [];	
    for (var i = 0; i < sequence.length; i++) {
		var w=get_window(sequence, i);
		var cmv=-Infinity
		if(w[1]-w[0]==window_size){		
			for (var j=w[0];j<w[1];j++){
				cmv=Math.max(cmv,window_scores[j]);	 
			}
		}
		scores.push(cmv); 		
		//scores.push(window_scores[i]);
    }
    return scores;
}    

function classify(sequence) {
    var window_propensities = window_scores(sequence);
    var scores;
    scores = super_window_scores(sequence, window_propensities);
    var max_score = Math.max.apply(Math,scores);
    var max_position = scores.indexOf(max_score);    
    //if max_score is None :
    //    max_score = -1.0
    //    max_position = -1
	
    return [max_score, max_position, scores];
}


function run() {
    for (var seq in exampleSequences) {
        var sequence = sequences[seq];
        //score,pos,scores,fold_indexes = classify(sequence)
        var ignore_fold_index = false;
        result = classify(sequence, ignore_fold_index);
        var score = result[0];
        var pos = result[1];
        var scores = result[2];        
        print(seq,score,pos);
    }
}

//run()
window.addEventListener('load', setup, false);

function byId(id) {
    return document.getElementById(id);
}

var sequenceTextArea;
var selectedSequenceList;
var resultTextArea;
var plotCanvas;
var plotContext;
function setup() {
    sequenceTextArea = byId("sequence");
    sequenceTextArea.addEventListener("input",update,false);
    resultTextArea = byId("result").firstChild;
    selectedSequenceList = byId("selectedSequence");
    selectedSequenceList.addEventListener("change",loadSelectedSequence,false);
    plotCanvas = byId("plot");
    plotContext = plotCanvas.getContext("2d");
}

function loadSelectedSequence(event) {
    var seqName = selectedSequenceList.selectedOptions[0].value;
    sequenceTextArea.value = exampleSequences[seqName].toUpperCase();
    update(null);
}

function update(event) {
    var seq = sequenceTextArea.value.toUpperCase();
    var ignore_fold_index = true;
    var result = classify(seq, ignore_fold_index);
    var score = result[0];
    var pos = result[1]+1;
    var scores = result[2];
    resultTextArea.nodeValue = "Max. Score = " + score.toFixed(3) + ",   Position = " + pos;
    // Erase canvas
    var w = plotCanvas.width;
    var h = plotCanvas.height;
    plotContext.fillStyle = "white";
    plotContext.fillRect(0,0,w,h);

    var noinf = scores.filter(function (s) { return s != -Infinity;});
    var maxScoreVal = Math.max.apply(Math,noinf.map(Math.abs));
    
    //console.log("before");
    //console.log("scores",scores);
   
    scores = scores.map(function(x) {return x/maxScoreVal;});
    
    //console.log("after");
    //console.log("scores",scores);
    
    function transform(x,y) {
        //return [(x-minx) / (maxx-minx) * w, h - (y-miny) / (maxy-miny) * h];
        return [x/seq.length * w, h - (y*0.95 + 1)/2  * h];  // y ranges from -1 to 1
    }        

    // x axis
    plotContext.beginPath();
    var p = transform(0,0);
    //console.log("moveTo",p);
    plotContext.moveTo(p[0],p[1]);
    p = transform(seq.length,0);
    //console.log("lineTo",p);
    plotContext.lineTo(p[0],p[1]);
    plotContext.lineWidth = 3;
    plotContext.strokeStyle = "gray";
    plotContext.stroke();

    // Position
    plotContext.beginPath();
    p = transform(pos+half_window_size,-1);
    plotContext.moveTo(p[0],p[1]);
    p = transform(pos+half_window_size,1);
    plotContext.lineTo(p[0],p[1]);
    plotContext.stroke();
    
    // Acids
    if (seq.length<100) {
        plotContext.fillStyle = "black";
        plotContext.textAlign = "center";
        plotContext.font = "8pt Arial";
        for (var i=0; i < seq.length; i++) {
            p = transform(i+1,0);
            p[1] += 10;
            plotContext.fillText(seq[i],p[0],p[1]);
        }
    }
    
    // scores
    // y axis
    plotContext.beginPath();
    var p = transform(0,-1);
    plotContext.moveTo(20+p[0],p[1]);
    p = transform(0,1);
    plotContext.lineTo(20+p[0],p[1]);
    plotContext.lineWidth = 2;
    plotContext.strokeStyle = "gray";
    plotContext.stroke();
    plotContext.fillStyle = "rgb(0,150,0)";
    plotContext.textAlign= "right";
    plotContext.font = "8pt Arial";
    for (var y = -1; y <= 1; y += 0.5) {
	p = transform(2,y);
	if (y == -1)
	    p[1] += -5;
	else if (y == 1)
	    p[1] += 5;
	plotContext.fillText((y*maxScoreVal).toFixed(1),p[0]+15,p[1]);
    }
    // scores curve
    var x0 = half_window_size;
    plotContext.beginPath();
    p = transform(x0+1,scores[0]);
    plotContext.moveTo(p[0],p[1]);
    for (var i=1; i < scores.length; i++) {
        p = transform(x0+i+1,scores[i]);
        //console.log(i,scores[i]);
        plotContext.lineTo(p[0],p[1]);
    }
    plotContext.strokeStyle = "rgba(0,150,0,0.9)";
    plotContext.lineWidth = 2;
    plotContext.stroke();

    // Labels
    plotContext.fillStyle = "rgb(150,0,0)";
    plotContext.textAlign = "left";
    plotContext.font = "bold 12pt Arial";
    plotContext.fillText("pRANK Score",20,h);  
}
