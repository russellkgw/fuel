class Data::Api::ExchangeRateFix
  def self.insert_averages
    missing_date = []
    exchange_rates = ExchangeRate.where("date >= '1995-11-01'").order(date: :asc).all.select { |x| x.date.cwday <= 5 }

    (Date.new(1995, 11, 1)..(Date.current - 1.day)).each do |date|
      next if date.cwday >= 6

      missing_date << date if exchange_rates.select { |x| x.date == date }.empty?
    end

    fixes = {}

    missing_date.each do |date|
      before = ExchangeRate.where("date < ?", date).order(date: :desc).first
      after = ExchangeRate.where("date > ?", date).order(date: :asc).first

      if before.present? && after.present?
        fixes[date.to_s] = (before.rate + after.rate) / 2.0
      end
    end

    fixes.each do |date, rate|
      next if ExchangeRate.where(date: date).any?

      ExchangeRate.create({ base: 'USD',
                            currency: 'ZAR',
                            rate: rate,
                            date: date,
                            source: 'AVERAGE-https://www.resbank.co.za',
                            created_at: DateTime.now,
                            updated_at: DateTime.now })
    end

    puts missing_date.to_s
    puts fixes.to_s
  end
end