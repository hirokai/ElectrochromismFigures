const width = 200;
const height = 150;

function mk_path(ps, x, y) {
    return "M " + x(ps[0][0]) + " " + y(ps[0][1]) + (_.map(ps, function (p) {
            return " L " + x(p[0]) + " " + y(p[1]);
        })).join("");
}

Vue.component('graph', {
    data: function () {
        const max_series = 10;
        const colors = d3.scaleOrdinal(d3.schemeCategory10);
        return {
            points: _.map(_.range(max_series), function () {
                return [[0, 0]];
            }),
            styleObject: {
                width: width, height: height
                , position: 'absolute'
                , top: +this.yi * (10 + height) + 100
                , left: +this.xi * (10 + width) + 10
            },
            colors: [colors(0), colors(1), colors(2)],
            num_series: 3
        }
    },
    template: '#graph-template',
    computed: {
        paths: function () {
            const y = d3.scaleLinear().range([height, 0]).domain([0, 70]);
            return _.map(this.points, function (ps) {
                const e = d3.extent(ps, function (p) {
                    return p[0];
                });
                const x = d3.scaleLinear().range([0, width]).domain([e[0], e[0] + 60]);
                return mk_path(ps, x, y);
            });
        },
        name: function () {
            return '' + this.pedot + ' perc, ' + this.voltage + ' V';
        },
        mode: function () {
            return this.voltage >= 0 ? 'ox' : 'red';
        }
    },
    mounted: function () {
        // const x = d3.scaleLinear().range([0, width]).domain(d3.extent(this.points, function (d) {
        //     return d[0];
        // }));
        // const svg = d3.select(this.$el, 'svg');
        // console.log(svg);
        // const axis = d3.axisBottom(x);
        // axis.ticks(10);
        // svg.append("g")
        //     .attr("transform", "translate(0," + 50 + ")")
        //     .call(axis);
        const self = this;
        _.map(['20160512-13', '20161013', '20161019'], function (d, i) {
            const url = './kinetics/' + (self.corrected ? 'split' : 'raw_split') +
                '/' + d + '/' + self.pedot + ' perc PEDOT - 2000 rpm/' + self.mode + ' ' + self.voltage + '.csv';
            if (self.voltage == 0.4 && self.pedot == '80') {
                console.log(url);
            }
            $.get(url, function (s) {
                const vs = _.map(d3.csvParseRows(s), function (row) {
                    return [parseFloat(row[0]), parseFloat(row[1])];
                });
                self.$set(self.points, i, vs);
            }).fail(function () {
                console.log('Not found: ' + url);
                app.missing.push(url);
                self.$set(self.points, i, [[0, 0], [0, 0]]);
            });
        });
        // _.map(['20160512-13', '20161013', '20161019'], function (d, i) {
        //     $.get('./kinetics/split/' + d + '/' + self.pedot + ' perc PEDOT - 2000 rpm/' + self.mode + ' ' + self.voltage + '.csv', function (s) {
        //         const vs = _.map(d3.csvParseRows(s), function (row) {
        //             return [parseFloat(row[0]), parseFloat(row[1])];
        //         });
        //         self.$set(self.points, i + 3, vs);
        //     });
        // });
    },
    props: ['voltage', 'pedot', 'xi', 'yi', 'corrected'],
    created: function () {
    }
});

Vue.component('pedot_voltage_graphs', {
    template: "#pedot_voltage_graphs-template",
    props: ['plot_mode', 'voltages', 'pedots', 'dates', 'corrected']
});

const app = new Vue({
    el: '#app',
    data: function () {
        const colors = d3.scaleOrdinal(d3.schemeCategory10);
        return {
            modes: ['red'] * 2 + ['ox'] * 5,
            voltages: [-0.5, -0.2, '0.0', 0.2, 0.4, 0.6, 0.8],
            pedots: [20, 30, 40, 60, 80],
            dates: ['0512-13', '1013', '1019'],
            colors: _.map(_.range(10), function (i) {
                return colors(i);
            }),
            plot_mode: 'pedot-voltage',
            corrected_mode: true,
            missing: []
        }
    },
    computed: {}
});

$('input[name="select-series"]').on('click', function () {
    const mode = $('input[name="select-series"]:checked').val();
    app.plot_mode = mode;
});

$('input#check-corrected').on('click', function (ev) {
    const mode = this.checked;
    app.corrected_mode = mode;
});