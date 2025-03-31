package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram count using "pairs" approach
 */
public class BigramFrequencyPairs extends Configured implements Tool {
        private static final Logger LOG = Logger.getLogger(BigramFrequencyPairs.class);

        /*
         * Mapper: emits <bigram, 1>, where bigram = (leftWord, rightWord)
         */
        private static class MyMapper extends
                        Mapper<LongWritable, Text, PairOfStrings, IntWritable> {

                // Reuse objects to save overhead of object creation.
                private static final IntWritable ONE = new IntWritable(1);
                private static final PairOfStrings BIGRAM = new PairOfStrings();

                @Override
                public void map(LongWritable key, Text value, Context context)
                                throws IOException, InterruptedException {
                        String line = ((Text) value).toString();
                        String[] words = line.trim().split("\\s+");
                        
                        if (words.length > 1) {
                                String previous_word = words[0];
                                for (int i = 1; i < words.length; i++) {
                                        String w = words[i];
                                        // Skip empty words
                                        if (w.length() == 0) {
                                                continue;
                                        }
                                        
                                        // Emit the bigram
                                        BIGRAM.set(previous_word, w);
                                        context.write(BIGRAM, ONE);
                                        
                                        // Also emit a special pair to count the left word
                                        BIGRAM.set(previous_word, "");
                                        context.write(BIGRAM, ONE);
                                        
                                        previous_word = w;
                                }
                                
                                // Don't forget the last word
                                BIGRAM.set(previous_word, "");
                                context.write(BIGRAM, ONE);
                        }
                }
        }

        /*
         * Reducer: aggregate bigram counts and compute relative frequencies
         */
        private static class MyReducer extends
                        Reducer<PairOfStrings, IntWritable, PairOfStrings, FloatWritable> {

                // Reuse objects.
                private final static FloatWritable VALUE = new FloatWritable();
                private Map<String, Integer> wordTotals = new HashMap<String, Integer>();
                private Map<String, Map<String, Integer>> wordBigrams = new HashMap<String, Map<String, Integer>>();

                @Override
                public void reduce(PairOfStrings key, Iterable<IntWritable> values,
                                Context context) throws IOException, InterruptedException {
                        // Sum up the counts for this bigram
                        int sum = 0;
                        for (IntWritable value : values) {
                                sum += value.get();
                        }
                        
                        String leftWord = key.getLeftElement();
                        String rightWord = key.getRightElement();
                        
                        // If this is a special pair to count the left word
                        if (rightWord.isEmpty()) {
                                // Store the total count for this word
                                if (wordTotals.containsKey(leftWord)) {
                                        wordTotals.put(leftWord, wordTotals.get(leftWord) + sum);
                                } else {
                                        wordTotals.put(leftWord, sum);
                                }
                        } else {
                                // Store the bigram count
                                if (!wordBigrams.containsKey(leftWord)) {
                                        wordBigrams.put(leftWord, new HashMap<String, Integer>());
                                }
                                wordBigrams.get(leftWord).put(rightWord, sum);
                        }
                }
                
                @Override
                protected void cleanup(Context context) throws IOException, InterruptedException {
                        PairOfStrings outputKey = new PairOfStrings();
                        
                        // Output the word totals and relative frequencies
                        for (String word : wordTotals.keySet()) {
                                int total = wordTotals.get(word);
                                
                                // Output the total count for this word
                                outputKey.set(word, "");
                                VALUE.set((float) total);
                                context.write(outputKey, VALUE);
                                
                                // Output the relative frequencies for each bigram
                                if (wordBigrams.containsKey(word)) {
                                        Map<String, Integer> bigrams = wordBigrams.get(word);
                                        for (String nextWord : bigrams.keySet()) {
                                                int count = bigrams.get(nextWord);
                                                float relFreq = (float) count / total;
                                                
                                                outputKey.set(word, nextWord);
                                                VALUE.set(relFreq);
                                                context.write(outputKey, VALUE);
                                        }
                                }
                        }
                }
        }
        
        private static class MyCombiner extends
                        Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
                private static final IntWritable SUM = new IntWritable();

                @Override
                public void reduce(PairOfStrings key, Iterable<IntWritable> values,
                                Context context) throws IOException, InterruptedException {
                        // Sum up the counts for this bigram
                        int sum = 0;
                        for (IntWritable value : values) {
                                sum += value.get();
                        }
                        SUM.set(sum);
                        context.write(key, SUM);
                }
        }

        /*
         * Partition bigrams based on their left elements
         */
        private static class MyPartitioner extends
                        Partitioner<PairOfStrings, IntWritable> {
                @Override
                public int getPartition(PairOfStrings key, IntWritable value,
                                int numReduceTasks) {
                        return (key.getLeftElement().hashCode() & Integer.MAX_VALUE)
                                        % numReduceTasks;
                }
        }

        /**
         * Creates an instance of this tool.
         */
        public BigramFrequencyPairs() {
        }

        private static final String INPUT = "input";
        private static final String OUTPUT = "output";
        private static final String NUM_REDUCERS = "numReducers";

        /**
         * Runs this tool.
         */
        @SuppressWarnings({ "static-access" })
        public int run(String[] args) throws Exception {
                Options options = new Options();

                options.addOption(OptionBuilder.withArgName("path").hasArg()
                                .withDescription("input path").create(INPUT));
                options.addOption(OptionBuilder.withArgName("path").hasArg()
                                .withDescription("output path").create(OUTPUT));
                options.addOption(OptionBuilder.withArgName("num").hasArg()
                                .withDescription("number of reducers").create(NUM_REDUCERS));

                CommandLine cmdline;
                CommandLineParser parser = new GnuParser();

                try {
                        cmdline = parser.parse(options, args);
                } catch (ParseException exp) {
                        System.err.println("Error parsing command line: "
                                        + exp.getMessage());
                        return -1;
                }

                // Lack of arguments
                if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
                        System.out.println("args: " + Arrays.toString(args));
                        HelpFormatter formatter = new HelpFormatter();
                        formatter.setWidth(120);
                        formatter.printHelp(this.getClass().getName(), options);
                        ToolRunner.printGenericCommandUsage(System.out);
                        return -1;
                }

                String inputPath = cmdline.getOptionValue(INPUT);
                String outputPath = cmdline.getOptionValue(OUTPUT);
                int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

                LOG.info("Tool: " + BigramFrequencyPairs.class.getSimpleName());
                LOG.info(" - input path: " + inputPath);
                LOG.info(" - output path: " + outputPath);
                LOG.info(" - number of reducers: " + reduceTasks);

                // Create and configure a MapReduce job
                Configuration conf = getConf();
                Job job = Job.getInstance(conf);
                job.setJobName(BigramFrequencyPairs.class.getSimpleName());
                job.setJarByClass(BigramFrequencyPairs.class);

                job.setNumReduceTasks(reduceTasks);

                FileInputFormat.setInputPaths(job, new Path(inputPath));
                FileOutputFormat.setOutputPath(job, new Path(outputPath));

                job.setMapOutputKeyClass(PairOfStrings.class);
                job.setMapOutputValueClass(IntWritable.class);
                job.setOutputKeyClass(PairOfStrings.class);
                job.setOutputValueClass(FloatWritable.class);

                /*
                 * A MapReduce program consists of three components: a mapper, a
                 * reducer, a combiner (which reduces the amount of shuffle data), and a partitioner
                 */
                job.setMapperClass(MyMapper.class);
                job.setCombinerClass(MyCombiner.class);
                job.setPartitionerClass(MyPartitioner.class);
                job.setReducerClass(MyReducer.class);

                // Delete the output directory if it exists already.
                Path outputDir = new Path(outputPath);
                FileSystem.get(conf).delete(outputDir, true);

                // Time the program
                long startTime = System.currentTimeMillis();
                job.waitForCompletion(true);
                LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
                                / 1000.0 + " seconds");

                return 0;
        }

        /**
         * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
         */
        public static void main(String[] args) throws Exception {
                ToolRunner.run(new BigramFrequencyPairs(), args);
        }
}
