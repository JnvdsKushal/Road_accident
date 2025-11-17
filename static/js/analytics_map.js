// Initialize the map (centered on Telangana)
var map = L.map("map").setView([17.5, 79.0], 7);

// Add the base tile layer
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "© OpenStreetMap contributors",
}).addTo(map);

// Define color-coded risk zones (Telangana & Hyderabad areas)
const riskZones = [
  // --- Hyderabad City Zones (High Risk) ---
  { area: "Secunderabad", coords: [17.4399, 78.4983], risk: "High", color: "red" },
  { area: "Ameerpet", coords: [17.4375, 78.4483], risk: "High", color: "red" },
  { area: "Kukatpally", coords: [17.4933, 78.3990], risk: "High", color: "red" },
  { area: "Madhapur", coords: [17.4477, 78.3914], risk: "High", color: "red" },
  { area: "Gachibowli", coords: [17.4401, 78.3489], risk: "High", color: "red" },
  { area: "Banjara Hills", coords: [17.4146, 78.4347], risk: "High", color: "red" },
  { area: "Jubilee Hills", coords: [17.4273, 78.4111], risk: "High", color: "red" },
  { area: "LB Nagar", coords: [17.3475, 78.5520], risk: "High", color: "red" },
  { area: "Dilsukhnagar", coords: [17.3687, 78.5246], risk: "High", color: "red" },
  { area: "Uppal", coords: [17.4001, 78.5596], risk: "High", color: "red" },
  { area: "Miyapur", coords: [17.4961, 78.3564], risk: "High", color: "red" },
  { area: "Chandanagar", coords: [17.4891, 78.3310], risk: "High", color: "red" },
  { area: "Tarnaka", coords: [17.4298, 78.5332], risk: "High", color: "red" },
  { area: "Begumpet", coords: [17.4422, 78.4738], risk: "High", color: "red" },
  { area: "Hitech City", coords: [17.4500, 78.3800], risk: "High", color: "red" },

  // --- Medium Risk Areas (Telangana towns & highways) ---
  { area: "Warangal", coords: [17.9784, 79.6002], risk: "Medium", color: "orange" },
  { area: "Karimnagar", coords: [18.4386, 79.1288], risk: "Medium", color: "orange" },
  { area: "Nizamabad", coords: [18.6725, 78.0941], risk: "Medium", color: "orange" },
  { area: "Khammam", coords: [17.2473, 80.1514], risk: "Medium", color: "orange" },
  { area: "Mahbubnagar", coords: [16.7480, 77.9850], risk: "Medium", color: "orange" },
  { area: "Nalgonda", coords: [17.0520, 79.2674], risk: "Medium", color: "orange" },
  { area: "Adilabad", coords: [19.6641, 78.5320], risk: "Medium", color: "orange" },
  { area: "Suryapet", coords: [17.1394, 79.6200], risk: "Medium", color: "orange" },
  { area: "Sangareddy", coords: [17.6231, 78.0817], risk: "Medium", color: "orange" },
  { area: "Medchal", coords: [17.6326, 78.4810], risk: "Medium", color: "orange" },
  { area: "Vikarabad", coords: [17.3360, 77.9042], risk: "Medium", color: "orange" },
  { area: "Jagtial", coords: [18.7895, 78.9121], risk: "Medium", color: "orange" },
  { area: "Mancherial", coords: [18.8716, 79.4435], risk: "Medium", color: "orange" },
  { area: "Ramagundam", coords: [18.7595, 79.4755], risk: "Medium", color: "orange" },
  { area: "Zaheerabad", coords: [17.6818, 77.6112], risk: "Medium", color: "orange" },
  { area: "Kodad", coords: [16.9946, 79.9656], risk: "Medium", color: "orange" },
  { area: "Bhongir", coords: [17.5155, 78.8885], risk: "Medium", color: "orange" },
  { area: "Siddipet", coords: [18.1019, 78.8486], risk: "Medium", color: "orange" },
  { area: "Kamareddy", coords: [18.3200, 78.3400], risk: "Medium", color: "orange" },
  { area: "Gadwal", coords: [16.2333, 77.8000], risk: "Medium", color: "orange" },
  { area: "Bhadrachalam", coords: [17.6673, 80.8881], risk: "Medium", color: "orange" },
  { area: "Kothagudem", coords: [17.5520, 80.6197], risk: "Medium", color: "orange" },
  { area: "Miryalaguda", coords: [16.8720, 79.5623], risk: "Medium", color: "orange" },
  { area: "Mahabubabad", coords: [17.6037, 80.0026], risk: "Medium", color: "orange" },
  { area: "Jangaon", coords: [17.7271, 79.1528], risk: "Medium", color: "orange" },
  { area: "Narayanpet", coords: [16.7456, 77.4969], risk: "Medium", color: "orange" },
  { area: "Tandur", coords: [17.2353, 77.5788], risk: "Medium", color: "orange" },
  { area: "Bhupalpally", coords: [18.4365, 79.9981], risk: "Medium", color: "orange" },
  { area: "Sircilla", coords: [18.3880, 78.8107], risk: "Medium", color: "orange" },
  { area: "Nagarkurnool", coords: [16.4801, 78.3106], risk: "Medium", color: "orange" },
  { area: "Wanaparthy", coords: [16.3624, 78.0674], risk: "Medium", color: "orange" },

  // --- Low Risk (Rural / low traffic areas) ---
  { area: "Asifabad", coords: [19.3580, 79.2840], risk: "Low", color: "green" },
  { area: "Nirmal", coords: [19.0960, 78.3440], risk: "Low", color: "green" },
  { area: "Kollapur", coords: [16.2425, 78.9970], risk: "Low", color: "green" },
  { area: "Utnoor", coords: [19.3662, 78.7911], risk: "Low", color: "green" },
  { area: "Banswada", coords: [18.3772, 77.8825], risk: "Low", color: "green" },
  { area: "Bellampalli", coords: [19.0638, 79.4932], risk: "Low", color: "green" },
  { area: "Kagaznagar", coords: [19.3319, 79.4661], risk: "Low", color: "green" },
  { area: "Bheemadevarpalle", coords: [18.1922, 79.4143], risk: "Low", color: "green" },
  { area: "Medak", coords: [18.0450, 78.2620], risk: "Low", color: "green" },
  { area: "Chevella", coords: [17.3113, 78.1321], risk: "Low", color: "green" },
  { area: "Pargi", coords: [17.1822, 77.9097], risk: "Low", color: "green" },
  { area: "Narayanakhed", coords: [17.8651, 77.7190], risk: "Low", color: "green" },
  { area: "Yellareddy", coords: [18.3465, 78.0845], risk: "Low", color: "green" },
  { area: "Husnabad", coords: [18.2015, 78.8821], risk: "Low", color: "green" },
  { area: "Bhainsa", coords: [19.1070, 77.9644], risk: "Low", color: "green" },
  { area: "Luxettipet", coords: [18.8754, 79.2272], risk: "Low", color: "green" },
  { area: "Tirumalagiri", coords: [17.6053, 79.6098], risk: "Low", color: "green" },
  { area: "Sadasivpet", coords: [17.6235, 77.9520], risk: "Low", color: "green" },
  { area: "Huzurabad", coords: [18.1954, 79.4002], risk: "Low", color: "green" },
  { area: "Choutuppal", coords: [17.3430, 78.9062], risk: "Low", color: "green" },
  { area: "Gajwel", coords: [17.8519, 78.6821], risk: "Low", color: "green" },
  { area: "Ibrahimpatnam", coords: [17.2056, 78.6022], risk: "Low", color: "green" },
  { area: "Mulugu", coords: [18.1935, 80.0444], risk: "Low", color: "green" },
  { area: "Narayanapur", coords: [17.2755, 78.9280], risk: "Low", color: "green" },
  { area: "Kalwakurthy", coords: [16.6610, 78.5101], risk: "Low", color: "green" },
  { area: "Pedapalli", coords: [18.6175, 79.3722], risk: "Low", color: "green" },
  { area: "Koheda", coords: [18.0750, 79.2200], risk: "Low", color: "green" },
  { area: "Raikal", coords: [18.8355, 78.8746], risk: "Low", color: "green" },
  { area: "Armur", coords: [18.7935, 78.2933], risk: "Low", color: "green" },
  { area: "Dubbak", coords: [18.1423, 78.6189], risk: "Low", color: "green" },
  { area: "Narayankhed", coords: [17.8600, 77.7160], risk: "Low", color: "green" },
];

// Add colored circles for each risk zone
riskZones.forEach((zone) => {
  L.circleMarker(zone.coords, {
    color: zone.color,
    fillColor: zone.color,
    fillOpacity: 0.6,
    radius: 10,
  })
    .addTo(map)
    .bindPopup(
      `<b>${zone.area}</b><br>Risk Level: <span style="color:${zone.color}; font-weight:bold;">${zone.risk}</span>`
    );
});

// Add a legend
const legend = L.control({ position: "bottomright" });
legend.onAdd = function () {
  const div = L.DomUtil.create("div", "legend");
  div.innerHTML += "<h4>Accident Risk Zones (Telangana)</h4>";
  div.innerHTML += '<i style="background:red"></i><span>High Risk</span><br>';
  div.innerHTML += '<i style="background:orange"></i><span>Medium Risk</span><br>';
  div.innerHTML += '<i style="background:green"></i><span>Low Risk</span><br>';
  return div;
};
legend.addTo(map);

// Legend CSS
const legendStyle = document.createElement("style");
legendStyle.innerHTML = `
  .legend {
    background: white;
    padding: 10px;
    border-radius: 8px;
    line-height: 22px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  }
  .legend i {
    width: 18px;
    height: 18px;
    float: left;
    margin-right: 8px;
    opacity: 0.8;
  }
`;
document.head.appendChild(legendStyle);
