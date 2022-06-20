import {Link} from 'react-router-dom'

function Landing() {
  return (
    <>
      <div className="container">
        <div className="mainSide">Sign Langauge Predictor</div>
        <div className="leftSide">
          <p>
            Hand gestures are one of the nonverbal communication modalities used
            in sign language. It is most often used by deaf people who have
            hearing or speech impairments to communicate with other deaf people
            or even with normal people, so a Deep Learning model is implemented
            to predict the signs and translate them accurately and that is our
            mission to help them. The model has been trained over a huge number
            of labelled videos and processed by Convolutional and LSTM layers.
            The dynamic visuals were extracted as points in three dimensions by
            MediaPipe
          </p>
        </div>
        <div className="rightSide">
            <Link to="/app" ><div className="button"></div></Link>          
        </div>
      </div>
    </>
  );
};

export default Landing;
