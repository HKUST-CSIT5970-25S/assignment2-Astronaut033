package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Compute the bigram count using "pairs" approach
 */
public class CORPairs extends Configured implements Tool {
        private static final Logger LOG = Logger.getLogger(CORPairs.class);

        /*
         * First-pass Mapper: emits <word, 1> for each word in a line
         */
        private static class CORMapper1 extends
                        Mapper<LongWritable, Text, Text, IntWritable> {
                @Override
                public void map(LongWritable key, Text value, Context context)
                                throws IOException, InterruptedException {
                        HashMap<String, Integer> word_set = new HashMap<String, Integer>();
                        // Please use this tokenizer! DO NOT implement a tokenizer by yourself!
                        String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
                        StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
                        
                        // Count each word in the line
                        while (doc_tokenizer.hasMoreTokens()) {
                                String word = doc_tokenizer.nextToken();
                                if (word.length() > 0) {  // Skip empty tokens
                                    if (word_set.containsKey(word)) {
                                            word_set.put(word, word_set.get(word) + 1);
                                    } else {
                                            word_set.put(word, 1);
                                    }
                                }
                        }
                        
                        // Emit each word and its count
                        for (Map.Entry<String, Integer> entry : word_set.entrySet()) {
                                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                        }
                }
        }

        /*
         * First-pass Reducer: sums up the counts for each word
         */
        private static class CORReducer1 extends
                        Reducer<Text, IntWritable, Text, IntWritable> {
                @Override
                public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
                        int sum = 0;
                        for (IntWritable value : values) {
                                sum += value.get();
                        }
                        context.write(key, new IntWritable(sum));
                }
        }


        /*
         * Second-pass Mapper: emits <word pair, 1> for each pair of words in a line
         */
        public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
                private static final IntWritable ONE = new IntWritable(1);
                private static final PairOfStrings PAIR = new PairOfStrings();
                
                @Override
                protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
                        // Please use this tokenizer! DO NOT implement a tokenizer by yourself!
                        StringTokenizer doc_tokenizer = new StringTokenizer(value.toString().replaceAll("[^a-z A-Z]", " "));
                        
                        // Collect all words in the line
                        Set<String> words = new HashSet<String>();
                        while (doc_tokenizer.hasMoreTokens()) {
                                String token = doc_tokenizer.nextToken();
                                if (token.length() > 0) {
                                        words.add(token);
                                }
                        }
                        
                        // Convert to sorted array for consistent ordering
                        String[] wordsArray = words.toArray(new String[0]);
                        Arrays.sort(wordsArray);
                        
                        // Emit each pair of words (all possible combinations)
                        for (int i = 0; i < wordsArray.length; i++) {
                                for (int j = 0; j < wordsArray.length; j++) {
                                        // Skip self-pairs
                                        if (i != j) {
                                                PAIR.set(wordsArray[i], wordsArray[j]);
                                                context.write(PAIR, ONE);
                                        }
                                }
                        }
                }
        }

        /*
         * Second-pass Combiner: sums up the counts for each word pair
         */
        private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
                private static final IntWritable SUM = new IntWritable();
                
                @Override
                protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
                        int sum = 0;
                        for (IntWritable value : values) {
                                sum += value.get();
                        }
                        SUM.set(sum);
                        context.write(key, SUM);
                }
        }

        /*
         * Second-pass Reducer: calculates COR(A, B) for each word pair
         */
        public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
                private final static Map<String, Integer> word_total_map = new HashMap<String, Integer>();
                private final static DoubleWritable COR_VALUE = new DoubleWritable();

                /*
                 * Preload the middle result file.
                 * In the middle result file, each line contains a word and its frequency Freq(A), seperated by "\t"
                 */
                @Override
                protected void setup(Context context) throws IOException, InterruptedException {
                        Path middle_result_path = new Path("mid/part-r-00000");
                        Configuration middle_conf = new Configuration();
                        try {
                                FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);

                                if (!fs.exists(middle_result_path)) {
                                        LOG.error("Middle result file does not exist: " + middle_result_path.toString());
                                        return;
                                }

                                FSDataInputStream in = fs.open(middle_result_path);
                                InputStreamReader inStream = new InputStreamReader(in);
                                BufferedReader reader = new BufferedReader(inStream);

                                LOG.info("Reading middle result file...");
                                String line = reader.readLine();
                                String[] line_terms;
                                int lineCount = 0;
                                while (line != null) {
                                        line_terms = line.split("\t");
                                        if (line_terms.length >= 2) {
                                                word_total_map.put(line_terms[0], Integer.valueOf(line_terms[1]));
                                                lineCount++;
                                        }
                                        line = reader.readLine();
                                }
                                reader.close();
                                LOG.info("Finished reading middle result file. Read " + lineCount + " words.");
                        } catch (Exception e) {
                                LOG.error("Error reading middle result file: " + e.getMessage());
                                e.printStackTrace();
                        }
                }

                /*
                 * Calculate COR(A, B) = Freq(A, B) / (Freq(A) * Freq(B))
                 */
                @Override
                protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
                        // Sum up the counts for this word pair
                        int pairFreq = 0;
                        for (IntWritable value : values) {
                                pairFreq += value.get();
                        }
                        
                        // Get the individual word frequencies
                        String wordA = key.getLeftElement();
                        String wordB = key.getRightElement();
                        
                        if (word_total_map.containsKey(wordA) && word_total_map.containsKey(wordB)) {
                                int freqA = word_total_map.get(wordA);
                                int freqB = word_total_map.get(wordB);
                                
                                if (freqA > 0 && freqB > 0) {
                                        // Calculate COR(A, B)
                                        double cor = (double) pairFreq / (freqA * freqB);
                                        COR_VALUE.set(cor);
                                        context.write(key, COR_VALUE);
                                }
                        }
                }
        }

        private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
                @Override
                public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
                        return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
                }
        }

        /**
         * Creates an instance of this tool.
         */
        public CORPairs() {
        }

        private static final String INPUT = "input";
        private static final String MIDDLE = "middle";
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
                String middlePath = "mid";
                String outputPath = cmdline.getOptionValue(OUTPUT);

                int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

                LOG.info("Tool: " + CORPairs.class.getSimpleName());
                LOG.info(" - input path: " + inputPath);
                LOG.info(" - middle path: " + middlePath);
                LOG.info(" - output path: " + outputPath);
                LOG.info(" - number of reducers: " + reduceTasks);

                // Setup for the first-pass MapReduce
                Configuration conf1 = new Configuration();

                Job job1 = Job.getInstance(conf1, "Firstpass");

                job1.setJarByClass(CORPairs.class);
                job1.setMapperClass(CORMapper1.class);
                job1.setReducerClass(CORReducer1.class);
                job1.setOutputKeyClass(Text.class);
                job1.setOutputValueClass(IntWritable.class);

                FileInputFormat.setInputPaths(job1, new Path(inputPath));
                FileOutputFormat.setOutputPath(job1, new Path(middlePath));

                // Delete the output directory if it exists already.
                Path middleDir = new Path(middlePath);
                FileSystem.get(conf1).delete(middleDir, true);

                // Time the program
                long startTime = System.currentTimeMillis();
                job1.waitForCompletion(true);
                LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime)
                                / 1000.0 + " seconds");

                // Setup for the second-pass MapReduce

                // Delete the output directory if it exists already.
                Path outputDir = new Path(outputPath);
                FileSystem.get(conf1).delete(outputDir, true);


                Configuration conf2 = new Configuration();
                Job job2 = Job.getInstance(conf2, "Secondpass");

                job2.setJarByClass(CORPairs.class);
                job2.setMapperClass(CORPairsMapper2.class);
                job2.setCombinerClass(CORPairsCombiner2.class);
                job2.setReducerClass(CORPairsReducer2.class);

                job2.setOutputKeyClass(PairOfStrings.class);
                job2.setOutputValueClass(DoubleWritable.class);
                job2.setMapOutputKeyClass(PairOfStrings.class);
                job2.setMapOutputValueClass(IntWritable.class);
                job2.setNumReduceTasks(reduceTasks);

                FileInputFormat.setInputPaths(job2, new Path(inputPath));
                FileOutputFormat.setOutputPath(job2, new Path(outputPath));

                // Time the program
                startTime = System.currentTimeMillis();
                job2.waitForCompletion(true);
                LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime)
                                / 1000.0 + " seconds");

                return 0;
        }

        /**
         * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
         */
        public static void main(String[] args) throws Exception {
                ToolRunner.run(new CORPairs(), args);
        }
}
