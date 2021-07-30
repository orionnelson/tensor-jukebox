import pytest
import warnings
warnings.filterwarnings('ignore')

@pytest.mark.speechrec
def testSpeechRec():
	from speechrec import speechrec
	testpath='test/wav'
	ts = speechrec.transcript(testpath + '/thisisatest.wav')
	correct = 'THIS IS A TEST'
	assert ts == correct, "Transcription Module Failed"


@pytest.mark.deepface
def testDeepFace():
	from DeepFaceExample import deepface
	testpath = 'test/img/testimage.jpg'
	output = deepface.checkinfo(testpath)
	cage=(28<=output[0]<=38)
	cgen= (output[1]=='Man')
	crace = (output[2]=='white')
	assert True==(cage and cgen and crace), "Deep Face Module Failed" 
