import "./App.css";
import Landing from "./Landing";
import Model from "./Model";
import { BrowserRouter as Router , Routes , Route} from 'react-router-dom'

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<Model />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
