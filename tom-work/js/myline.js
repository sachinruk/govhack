

(function (_this, $) {

   var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

    var x = d3.scale.linear()
        .range([0, width]);

    var y = d3.scale.linear()
        .range([height, 0]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom")
        .tickFormat(d3.format("d"));

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left");

    var line = d3.svg.line()
        .x(function(d) { return x(d.year); })
        .y(function(d) { return y(d.numdeaths); });

    var svg = d3.select(".lineGraph").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var initialSetupPerformed = false;

    _this.initialise = function () {
        $(function () {
            // LineGraph.setupLineGraph('data/deaths_0.tsv');
        });

        return _this;
    }

    _this.removeLines = function() {
        d3.select('.lineGraph .line').remove();
        d3.select('.lineGraph .x.axis').remove();
        d3.select('.lineGraph .y.axis').remove();
        svg.datum();
    }


    _this.setupLineGraph = function(filename, extraClass) {
        d3.tsv(filename, function(error, data) {
          if (error) throw error;

          data.forEach(function(d) {
            d.year = d.year;
            d.numdeaths = d.numdeaths;
          });

          x.domain(d3.extent(data, function(d) { return d.year; }));
          y.domain(d3.extent(data, function(d) { return d.numdeaths; }));

          svg.datum(data);

          //if(!initialSetupPerformed) {
            svg.append("g")
              .attr("class", "x axis")
              .attr("transform", "translate(0," + height + ")")
              .call(xAxis);

            svg.append("g")
              .attr("class", "y axis")
              .call(yAxis);

            //initialSetupPerformed = true;
          //}       

          svg.append("path")
              .datum(data)
              .attr("class", ("line" + extraClass))
              .attr("d", line);
        });
    }

    // Initialise & assign to global scope
    window.LineGraph = _this.initialise();
})(window.LineGraph || {}, jQuery);